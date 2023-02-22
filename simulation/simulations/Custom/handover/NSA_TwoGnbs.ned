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
package simu5g.simulations.Custom.handover;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.eNodeB;
import simu5g.nodes.NR.gNodeB;
import simu5g.nodes.NR.NRUe;
import simu5g.nodes.PgwStandard;
import simu5g.nodes.backgroundCell.BackgroundCell;
import simu5g.world.radio.LteChannelControl;

// SingleCell_withSecondaryGnb
//
// This network represents the deployment option 3a in 3GPP 38.801
// A gNodeB is serving as secondary node for a LTE eNodeB, which is connected
// to the LTE EPC. All data-plane traffic goes through the LTE eNodeB 
//
network NSA_TwoGnbs
{
    parameters:
        int numUe = default(1);
        int numBgCells = default(0);
        @display("i=block/network2;bgb=1000,600;bgi=background/pisa");
    submodules:
        channelControl: LteChannelControl {
            @display("p=69.65,24.875;is=s");
        }
        routingRecorder: RoutingTableRecorder {
            @display("p=70.89375,85.81875;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=70.89375,152.98125;is=s");
        }
        binder: Binder {
            @display("p=70.89375,230.09375;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=70.89375,289.79376;is=s");
        }
        server: StandardHost {
            @display("p=210.19374,63.431248;is=n;i=device/server");
        }
        router: Router {
            @display("p=361.93124,62.1875;i=device/smallrouter");
        }
        pgw: PgwStandard {
            @display("p=524.8625,63.431248;is=l");
        }
        masterEnb: eNodeB {
            @display("p=524.8625,242.53125;is=vl");
        }
        secondaryGnb1: gNodeB {
            @display("p=319.64374,325.8625;is=vl");
        }
        secondaryGnb2: gNodeB {
            @display("p=726.35,325.8625;is=vl");
        }
        bgCell[numBgCells]: BackgroundCell {
            @display("p=910.425,98.25625;is=vl");
        }
        ue[numUe]: NRUe {
            @display("p=409.12997,470.625");
        }
    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> pgw.filterGate;
        pgw.pppg++ <--> Eth10G <--> masterEnb.ppp;

        //# X2 connections
        masterEnb.x2++ <--> Eth10G <--> secondaryGnb1.x2++;
        masterEnb.x2++ <--> Eth10G <--> secondaryGnb2.x2++;
        secondaryGnb1.x2++ <--> Eth10G <--> secondaryGnb2.x2++;
}
