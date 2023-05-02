//
//                  Simu5G
//
// Authors: Giovanni Nardini, Giovanni Stea, Antonio Virdis (University of Pisa)
//
// This file is part of a software released under the license included in file
// "license.pdf". Please read LICENSE and README files before using it.
// The above files and the present reference are part of the software itself,
// and cannot be removed from it.
//

#include "stack/phy/layer/NRPhyUe.h"
#include "stack/ip2nic/IP2Nic.h"
#include "stack/phy/feedback/LteDlFeedbackGenerator.h"
#include "stack/d2dModeSelection/D2DModeSelectionBase.h"
#include <fstream>

Define_Module(NRPhyUe);

NRPhyUe::NRPhyUe()
{
    handoverStarter_ = NULL;
    handoverTrigger_ = NULL;
}

NRPhyUe::~NRPhyUe()
{
}

void NRPhyUe::initialize(int stage)
{
    LtePhyUeD2D::initialize(stage);
    if (stage == inet::INITSTAGE_LOCAL)
    {
        isNr_ = (strcmp(getFullName(),"nrPhy") == 0) ? true : false;
        if (isNr_)
            otherPhy_ = check_and_cast<NRPhyUe*>(getParentModule()->getSubmodule("phy"));
        else
            otherPhy_ = check_and_cast<NRPhyUe*>(getParentModule()->getSubmodule("nrPhy"));
    }
}

void NRPhyUe::handleAirFrame(cMessage* msg)
{
    connectedNodeId_ = masterId_;

    if (useBattery_)
    {
        //TODO BatteryAccess::drawCurrent(rxAmount_, 0);
    }

    UserControlInfo* lteInfo = check_and_cast<UserControlInfo*>(msg->removeControlInfo());
    LteAirFrame* frame = check_and_cast<LteAirFrame*>(msg);

    EV << "NRPhyUe: received new LteAirFrame with ID " << frame->getId() << " from channel" << endl;

    // Source node has left the simulation
    if(binder_->getOmnetId(lteInfo->getSourceId()) == 0)
    {
        delete msg;
        return;
    }

    double carrierFreq = lteInfo->getCarrierFrequency();
    LteChannelModel* channelModel = getChannelModel(carrierFreq);

    // Carrier frequency not supported by this UE
    if (channelModel == NULL)
    {
        EV << "Received packet on carrier frequency not supported by this node. Delete it." << endl;
        delete lteInfo;
        delete frame;
        return;
    }

    // Handle handover frame
    if (lteInfo->getFrameType() == HANDOVERPKT)
    {
        // The message is on another carrier frequency or handover is already in process
        if (carrierFreq != primaryChannelModel_->getCarrierFrequency() || (handoverTrigger_ != NULL && handoverTrigger_->isScheduled()))
        {
            EV << "Received handover packet on a different carrier frequency. Delete it." << endl;
            delete lteInfo;
            delete frame;
            return;
        }

        // The message is from a different cellular technology
        if (lteInfo->isNr() != isNr_)
        {
            EV << "Received handover packet [from NR=" << lteInfo->isNr() << "] from a different radio technology [to NR=" << isNr_ << "]. Delete it." << endl;
            delete lteInfo;
            delete frame;
            return;
        }

        // The source node is a secondary node and the other stack of this UE is not attached to its master node
        // In this case, the UE cannot attach/communicate with this secondary node and the packet must be deleted.
        int sourceId = lteInfo->getSourceId();
        MacNodeId masterNodeId = binder_->getMasterNode(sourceId);
        if (sourceId != masterNodeId && otherPhy_->getMasterId() != masterNodeId)
        {
            EV << "Received handover packet from " << sourceId << ", which is a secondary node to a master [" << masterNodeId << "] different from the one this UE is attached to. Delete packet." << endl;
            delete lteInfo;
            delete frame;
            return;
        }

        handoverHandler(frame, lteInfo);
        return;
    }

    // The frame is not for this UE and this is not a multicast communication where this UE is enrolled in its target multicast group
    if (lteInfo->getDestId() != nodeId_ && !(binder_->isInMulticastGroup(nodeId_, lteInfo->getMulticastGroupId())))
    {
        EV << "ERROR: Frame is not for this UE. Delete it." << endl;
        EV << "Packet Type: " << phyFrameTypeToA((LtePhyFrameType)lteInfo->getFrameType()) << endl;
        EV << "Frame MacNodeId: " << lteInfo->getDestId() << endl;
        EV << "Local MacNodeId: " << nodeId_ << endl;
        delete lteInfo;
        delete frame;
        return;
    }

    /*
     * The UE associates with a new master while a packet from the old master is in the air
     * Event timing: TTI x: packet scheduled and sent to the UE (tx time = 1ms)
     *               TTI x+0.1: UE changes master
     *               TTI x+1: packet from the old master arrives at the UE
     */
    if (lteInfo->getDirection() != D2D && lteInfo->getDirection() != D2D_MULTI && lteInfo->getSourceId() != masterId_)
    {
        EV << "WARNING: frame to a UE that is leaving this cell (handover): deleted " << endl;
        EV << "Source MacNodeId: " << lteInfo->getSourceId() << endl;
        EV << "UE MacNodeId: " << nodeId_ << endl;
        delete lteInfo;
        delete frame;
        return;
    }

    // HACK: if this is a multicast connection, change the destId of the airframe so that upper layers can handle it
    if (binder_->isInMulticastGroup(nodeId_, lteInfo->getMulticastGroupId()))
    {
        lteInfo->setDestId(nodeId_);
    }

    // Send H-ARQ feedback up
    if (lteInfo->getFrameType() == HARQPKT || lteInfo->getFrameType() == GRANTPKT || lteInfo->getFrameType() == RACPKT || lteInfo->getFrameType() == D2DMODESWITCHPKT)
    {
        handleControlMsg(frame, lteInfo);
        return;
    }

    /* This is a DATA packet */

    // If the packet is a D2D multicast one, store it and decode it at the end of the TTI
    if (d2dMulticastEnableCaptureEffect_ && binder_->isInMulticastGroup(nodeId_, lteInfo->getMulticastGroupId()))
    {
        // If not already started, auto-send a message to signal the presence of data to be decoded
        if (d2dDecodingTimer_ == NULL)
        {
            d2dDecodingTimer_ = new cMessage("d2dDecodingTimer");
            d2dDecodingTimer_->setSchedulingPriority(10); // last thing to be performed in this TTI
            scheduleAt(NOW, d2dDecodingTimer_);
        }

        // Store frame together with related control info
        frame->setControlInfo(lteInfo);
        storeAirFrame(frame); // implements the capture effect

        return; // exit the function, decoding will be done later
    }

    // Compute DL CQI
    if (lteInfo->getUserTxParams() != NULL && lteInfo->getDirection() == DL)
    {
        int cw = (lteInfo->getUserTxParams()->readCqiVector().size() == 1) ? 0 : lteInfo->getCw();
        double cqi = lteInfo->getUserTxParams()->readCqiVector()[cw];

        emit(averageCqiDl_, cqi);
        recordCqi(cqi, DL);

        /* Emit UE statistics */
        checkEnbs();

        emit(positionX_, masterPosX_);
        emit(positionY_, masterPosY_);
        emit(servingRSRP_, masterRsrp_);
        emit(servingRSRQ_, masterRsrq_);
        emit(servingSINR_, masterSinr_);
        emit(servingDistance_, masterDistance_);
        emit(neighborTop1RSRP_, neighborRSRP_[0]);
        emit(neighborTop2RSRP_, neighborRSRP_[1]);
        emit(neighborTop1SINR_, neighborSINR_[0]);
        emit(neighborTop2SINR_, neighborSINR_[1]);
        emit(neighborTop1Distance_, neighborDistance_[0]);
        emit(neighborTop2Distance_, neighborDistance_[1]);
        emit(servingCell_, (long)masterId_);
        emit(timestamp_,simTime());
        emit(UEid_,nodeId_);
    }

    // Apply decider to received packet
    bool result;
    RemoteSet r = lteInfo->getUserTxParams()->readAntennaSet();
    if (r.size() > 1)
    {
        // DAS
        for (RemoteSet::iterator it = r.begin(); it != r.end(); it++)
        {
            EV << "NRPhyUe: Receiving Packet from antenna " << (*it) << "\n";

            /*
             * On UE: set the sender position and tx power to the sender das antenna
             */

            // cc->updateHostPosition(myHostRef,das_->getAntennaCoord(*it));
            // Set position of sender
            // Move m;
            // m.setStart(das_->getAntennaCoord(*it));

            RemoteUnitPhyData data;
            data.txPower = lteInfo->getTxPower();
            data.m = getRadioPosition();
            frame->addRemoteUnitPhyDataVector(data);
        }

        // apply analog models For DAS
        result = channelModel->isErrorDas(frame, lteInfo);
    }
    else
    {
        result = channelModel->isError(frame, lteInfo);
    }

    // Update statistics
    if (result)
        numAirFrameReceived_++;
    else
        numAirFrameNotReceived_++;

    EV << "Handled LteAirframe with ID " << frame->getId() << " with result "
       << (result ? "RECEIVED" : "NOT RECEIVED") << endl;

    auto pkt = check_and_cast<inet::Packet *>(frame->decapsulate());

    // Frame is no longer useful, destroy it
    delete frame;

    // Attach the decider result to the packet as control info
    lteInfo->setDeciderResult(result);
    *(pkt->addTagIfAbsent<UserControlInfo>()) = *lteInfo;
    delete lteInfo;

    // Send decapsulated message along with result control info to upperGateOut_
    send(pkt, upperGateOut_);

    if (getEnvir()->isGUI())
        updateDisplayString();
}

void NRPhyUe::triggerHandover()
{
    std::ofstream logfile;
    logfile.open("/home/kvasir/Documents/Ericsson_Project/Omnetpp/run.log",  std::ios_base::app);

    EV << "####Handover starting:####" << endl;
    EV << "current master: " << masterId_ << endl;
    EV << "current rssi: " << currentMasterRssi_ << endl;
    EV << "candidate master: " << candidateMasterId_ << endl;
    EV << "candidate rssi: " << candidateMasterRssi_ << endl;
    EV << "############" << endl;

    logfile << "####Handover starting:####" << endl;
    logfile << "current master: " << masterId_ << endl;
    logfile << "current rssi: " << currentMasterRssi_ << endl;
    logfile << "candidate master: " << candidateMasterId_ << endl;
    logfile << "candidate rssi: " << candidateMasterRssi_ << endl;
    logfile << "############" << endl;

    // The candidate node is a secondary node and the other stack of this UE is attached to its master node
    MacNodeId masterNodeId = binder_->getMasterNode(candidateMasterId_);
    if (candidateMasterId_ != masterNodeId && otherPhy_->getMasterId() == masterNodeId)
    {
        const std::pair<MacNodeId, MacNodeId>* handoverPair = binder_->getHandoverTriggered(otherPhy_->getMacNodeId());

        // The other stack is performing a handover
        if (handoverPair != NULL)
        {
            if (handoverPair->second == candidateMasterId_)
            {
                // Delay this handover
                double delta = handoverDelta_;

                // the other stack is performing a complete handover
                if (handoverPair->first != 0)
                    delta += handoverDetachment_+handoverAttachment_;
                // the other stack is attaching to an eNodeB
                else
                    delta += handoverAttachment_;

                // Wait for the other stack to complete its handover
                scheduleAt(simTime() + delta, handoverStarter_);
                EV << NOW << " NRPhyUe::triggerHandover - Wait the handover completion for the other stack. Delay this handover." << endl;
                logfile << NOW << " NRPhyUe::triggerHandover - Wait the handover completion for the other stack. Delay this handover." << endl;
            }
            else
            {
                // cancel this handover since the master is performing handover
                binder_->removeHandoverTriggered(nodeId_);
                EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is canceling its handover to eNB " << candidateMasterId_ << " since the master is performing handover" << endl;
                logfile << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is canceling its handover to eNB " << candidateMasterId_ << " since the master is performing handover" << endl;
            }

            return;
        }
    }

    // The other stack is connected to a node which is a secondary node of master node masterId_
    if (otherPhy_->getMasterId() != 0 && binder_->getMasterNode(otherPhy_->getMasterId()) == masterId_)
    {
        EV << NOW << " NRPhyUe::triggerHandover - Forcing detachment from " << otherPhy_->getMasterId() << " which is a secondary node to " << masterId_ << ". Delay this handover." << endl;
        logfile << NOW << " NRPhyUe::triggerHandover - Forcing detachment from " << otherPhy_->getMasterId() << " which is a secondary node to " << masterId_ << ". Delay this handover." << endl;

        // need to wait for the other stack to complete detachment
        scheduleAt(simTime() + handoverDetachment_+handoverDelta_, handoverStarter_);

        // Trigger detachment (handover to node 0)
        otherPhy_->forceHandover();

        return;
    }

    double handoverLatency;

    // Attachment only
    if (masterId_ == 0) {
        handoverLatency = handoverAttachment_;
        EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting attachment procedure to eNB " << candidateMasterId_ << "... " << endl;
        logfile << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting attachment procedure to eNB " << candidateMasterId_ << "... " << endl;
    }
    // Detachment only
    else if (candidateMasterId_ == 0) {
        handoverLatency = handoverDetachment_;
        EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " lost its connection to eNB " << masterId_ << ". Now detaching... " << endl;
        logfile << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " lost its connection to eNB " << masterId_ << ". Now detaching... " << endl;
    }
    // Complete handover
    else {
        handoverLatency = handoverDetachment_ + handoverAttachment_;
        EV << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting handover to eNB " << candidateMasterId_ << "... " << endl;
        logfile << NOW << " NRPhyUe::triggerHandover - UE " << nodeId_ << " is starting handover to eNB " << candidateMasterId_ << "... " << endl;
    }

    binder_->addUeHandoverTriggered(nodeId_);

    // Inform the UE's IP2Nic module to start holding downstream packets
    IP2Nic* ip2nic = check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->triggerHandoverUe(candidateMasterId_, isNr_);


    if (masterId_ != 0)
    {
        cModule* enb = getSimulation()->getModule(binder_->getOmnetId(masterId_));

        // Inform the eNB's IP2Nic module to forward data to the target eNB
        if (candidateMasterId_ != 0)
        {
            logfile << "Trigger IP2Nic mode switch\n";
            IP2Nic* enbIp2nic = check_and_cast<IP2Nic*>(enb->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
            enbIp2nic->triggerHandoverSource(nodeId_, candidateMasterId_);
        }

        // stop active D2D flows (go back to Infrastructure mode)
        // currently, DM is possible only for UEs served by the same cell

        // trigger D2D mode switch
        logfile << "Trigger D2d mode switch\n";
        D2DModeSelectionBase *d2dModeSelection = check_and_cast<D2DModeSelectionBase*>(enb->getSubmodule("cellularNic")->getSubmodule("d2dModeSelection"));
        logfile << "Cleared the trigger D2d mode switch\n";
        d2dModeSelection->doModeSwitchAtHandover(nodeId_, false);
        logfile << "Cleared final part of trigger D2d mode switch\n";
    }

    handoverTrigger_ = new cMessage("handoverTrigger");
    scheduleAt(simTime()+handoverLatency, handoverTrigger_);
}

void NRPhyUe::doHandover()
{
    std::ofstream logfile;
    logfile.open("/home/kvasir/Documents/Ericsson_Project/Omnetpp/run.log",  std::ios_base::app);
    logfile << "Handover process started " << simTime() << "\n";
    
    // change masterId
    MacNodeId oldMasterId = masterId_;
    masterId_ = candidateMasterId_;

    // change MasterRssi and hysteresisTh
    currentMasterRssi_ = candidateMasterRssi_;
    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);

    binder_->updateUeInfoCellId(nodeId_, masterId_);
    logfile << "Right before parent module if\n";

    // @author Alessandro Noferi
    if(getParentModule()->getParentModule()->findSubmodule("NRueCollector") != -1)
    {
        binder_->moveUeCollector(nodeId_, oldMasterId, masterId_);
    }
    logfile << "Right after parent module if\n";

    // do MAC operations for handover
    mac_->doHandover(masterId_);

    // When the UE was attached to an old eNodeB, it has to perform detachment procedures
    if (oldMasterId != 0)
    {
        logfile << "Entered detachment if\n";
        // Delete Old Buffers
        deleteOldBuffers(oldMasterId);

        // amc calls
        LteAmc *oldAmc = getAmcModule(oldMasterId);
        oldAmc->detachUser(nodeId_, UL);
        oldAmc->detachUser(nodeId_, DL);
        oldAmc->detachUser(nodeId_, D2D);

        // binder calls
        binder_->unregisterNextHop(oldMasterId, nodeId_);

        cellInfo_->detachUser(nodeId_);
    }

    logfile << "Cleared detachment if\n";

    // When the UE is attaching to a new eNodeB, it has to perform attachment procedures
    if (masterId_ != 0)
    {
        logfile << "Entered second if\n";
        
        LteAmc *newAmc = getAmcModule(masterId_);
        newAmc->attachUser(nodeId_, UL);
        newAmc->attachUser(nodeId_, DL);
        newAmc->attachUser(nodeId_, D2D);

        // binder calls
        binder_->registerNextHop(masterId_, nodeId_);
        das_->setMasterRuSet(masterId_);

        // update cellInfo
        cellInfo_ = check_and_cast<LteMacEnb*>(getSimulation()->getModule(binder_->getOmnetId(masterId_))
                ->getSubmodule("cellularNic")->getSubmodule("mac"))->getCellInfo();
        cellInfo_->attachUser(nodeId_);

        // first time the UE is attached to someone
        if (oldMasterId == 0)
        {
            int index = intuniform(0, binder_->phyPisaData.maxChannel() - 1);
            cellInfo_->lambdaInit(nodeId_, index);
            cellInfo_->channelUpdate(nodeId_, intuniform(1, binder_->phyPisaData.maxChannel2()));
        }

        // send a self-message to schedule the possible mode switch at the end of the TTI (after all UEs have performed the handover)
        cMessage* msg = new cMessage("doModeSwitchAtHandover");
        msg->setSchedulingPriority(10);
        scheduleAt(NOW, msg);
    }

    // update DL feedback generator
    LteDlFeedbackGenerator* fbGen;
    if (!isNr_)
        fbGen = check_and_cast<LteDlFeedbackGenerator*>(getParentModule()->getSubmodule("dlFbGen"));
    else
        fbGen = check_and_cast<LteDlFeedbackGenerator*>(getParentModule()->getSubmodule("nrDlFbGen"));
    fbGen->handleHandover(masterId_);

    // collect stat
    //emit(servingCell_, (long)masterId_);

    if (masterId_ == 0)
        EV << NOW << " NRPhyUe::doHandover - UE " << nodeId_ << " detached from the network" << endl;
    else
        EV << NOW << " NRPhyUe::doHandover - UE " << nodeId_ << " has completed handover to eNB " << masterId_ << "... " << endl;

    binder_->removeUeHandoverTriggered(nodeId_);
    binder_->removeHandoverTriggered(nodeId_);

    // inform the UE's IP2Nic module to forward held packets
    IP2Nic* ip2nic =  check_and_cast<IP2Nic*>(getParentModule()->getSubmodule("ip2nic"));
    ip2nic->signalHandoverCompleteUe(isNr_);

    // inform the eNB's IP2Nic module to forward data to the target eNB
    if (oldMasterId != 0 && masterId_ != 0)
    {
        IP2Nic* enbIp2nic =  check_and_cast<IP2Nic*>(getSimulation()->getModule(binder_->getOmnetId(masterId_))->getSubmodule("cellularNic")->getSubmodule("ip2nic"));
        enbIp2nic->signalHandoverCompleteTarget(nodeId_, oldMasterId);
    }

    logfile.close();
}

void NRPhyUe::forceHandover(MacNodeId targetMasterNode, double targetMasterRssi)
{
    Enter_Method_Silent();
    candidateMasterId_ = targetMasterNode;
    candidateMasterRssi_ = targetMasterRssi;
    hysteresisTh_ = updateHysteresisTh(currentMasterRssi_);

    cancelEvent(handoverStarter_);  // if any
    scheduleAt(NOW, handoverStarter_);
}

void NRPhyUe::deleteOldBuffers(MacNodeId masterId)
{
    /* Delete Mac Buffers */

    // delete macBuffer[nodeId_] at old master
    LteMacEnb *masterMac = check_and_cast<LteMacEnb *>(getMacByMacNodeId(masterId));
    masterMac->deleteQueues(nodeId_);

    // delete queues for master at this ue
    mac_->deleteQueues(masterId_);

    /* Delete Rlc UM Buffers */

    // delete UmTxQueue[nodeId_] at old master
    LteRlcUm *masterRlcUm = check_and_cast<LteRlcUm*>(getRlcByMacNodeId(masterId, UM));
    masterRlcUm->deleteQueues(nodeId_);

    // delete queues for master at this ue
    rlcUm_->deleteQueues(nodeId_);

    /* Delete PDCP Entities */

    // delete pdcpEntities[nodeId_] at old master
    // in case of NR dual connectivity, the master can be a secondary node, hence we have to delete PDCP entities residing the node's master
    MacNodeId masterNodeId = binder_->getMasterNode(masterId);
    LtePdcpRrcEnb* masterPdcp = check_and_cast<LtePdcpRrcEnb *>(getPdcpByMacNodeId(masterNodeId));
    masterPdcp->deleteEntities(nodeId_);

    // delete queues for master at this ue
    pdcp_->deleteEntities(masterId_);
}
