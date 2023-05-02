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
package simu5g.simulations.Custom.generate;

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
network Network
{
    parameters:
        int numUe = default(1);
        int numBgCells = default(0);
        @display("i=block/network2;bgb=480,250;bgi=background/pisa");
    submodules:
        channelControl: LteChannelControl {
            @display("p=30.040968,12.289487;is=s");
        }
        routingRecorder: RoutingTableRecorder {
            @display("p=30.496134,40.054623;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=30.496134,71.91626;is=s");
        }
        binder: Binder {
            @display("p=30.496134,97.860725;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=30.496134,127.79376;is=s");
        }
        server: StandardHost {
            @display("p=207.10062,21.39281;is=n;i=device/server");
        }
        router: Router {
            @display("p=237.59674,20.937645;i=device/smallrouter");
        }
        pgw: PgwStandard {
            @display("p=266,21;is=l");
        }
        masterEnb: eNodeB {
            @display("p=266,110;is=vl");
        }
        gnb152: gNodeB {
            @display("p=413,27;is=vl");
        }
        gnb425: gNodeB {
            @display("p=396,190;is=vl");
        }
        gnb508: gNodeB {
            @display("p=138,44;is=vl");
        }
        gnb686: gNodeB {
            @display("p=275,194;is=vl");
        }
        gnb907: gNodeB {
            @display("p=101,198;is=vl");
        }
//        bgCell[numBgCells]: BackgroundCell {
//            @display("p=235,120;is=vl");
//        }
        ue[numUe]: NRUe {
            @display("p=80,200");
        }
    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> pgw.filterGate;
        pgw.pppg++ <--> Eth10G <--> masterEnb.ppp;

        //# X2 connections
        masterEnb.x2++ <--> Eth10G <--> gnb152.x2++;
        masterEnb.x2++ <--> Eth10G <--> gnb425.x2++;
        masterEnb.x2++ <--> Eth10G <--> gnb508.x2++;
        masterEnb.x2++ <--> Eth10G <--> gnb686.x2++;
        masterEnb.x2++ <--> Eth10G <--> gnb907.x2++;
}
