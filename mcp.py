#!/usr/bin/python
# -*- coding: latin-1 -*-

import os, sys, platform
import pyodbc
import numpy as np
import scipy.io      as sio
import matplotlib.pyplot as plt
import pandas
from operator import itemgetter

import inspect
import os.path                  as ospath
import pickle
import cPickle

from datetime import datetime


def main():

    currentFilePath = ospath.dirname(inspect.getfile(inspect.currentframe()))

    loadFile = currentFilePath + '/DemandData.csv'


    scedFile = ospath.join(ospath.dirname(__file__), 'SCED_Data.csv')

    assert ospath.exists(loadFile)

    tempFile = ospath.join(currentFilePath,'Fig')
    if not ospath.exists(tempFile):
        os.makedirs(tempFile)
    
    tempFile = ospath.join(currentFilePath,'Output')
    if not ospath.exists(tempFile):
        os.makedirs(tempFile)
    dataPath = ospath.join(currentFilePath, 'Output')

    print("read load file")
    df = pandas.read_csv(loadFile);
    columnnames = list(df.columns.values)
    print columnnames
    tempLoad = df.values
    loadTimeStamp1 = tempLoad[:, 0]

    hrLoad = tempLoad[:, 1]
    print("read SCED file")

    df = pandas.read_csv(scedFile);

    columnnames = list(df.columns.values)
    print columnnames



    tempSCED = df.values
    scedTimeStamp1=tempSCED[:,0]

    loadTimeStamp = sorted([datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in loadTimeStamp1])

    sorted_inds,scedTimeStamp = zip(*sorted([(e,datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")) for e,dt in enumerate(scedTimeStamp1)], key=itemgetter(1)))

    used = set()
    resourceNameList=tempSCED[sorted_inds, 1]

    resourceNameArr = [(x.encode('UTF8')) for x in resourceNameList if x not in used and (used.add(x) or True)]
    resourceNameUnique = np.asarray(resourceNameArr)

    used = set()
    resourceTypeList = tempSCED[sorted_inds, 2]
    resourceTypeArr = [(x.encode('UTF8')) for x in resourceTypeList if x not in used and (used.add(x) or True)]
    resourceTypeUnique = np.asarray(resourceTypeArr)

    output = open(dataPath+'/resourceNameUnique.pkl', 'wb')
    pickle.dump(resourceNameUnique, output)
    output.close()

    bidPair =tempSCED[sorted_inds,3:]

    #-------------------------------end data reading--------------------------

    print ("\nComplete Input data reading.")

    # -------------------------reorganize bidding data based on the hour/resource -----------------------------

    numHour = len(loadTimeStamp)

    numResource = len(resourceNameUnique)

    numYear = 1
    parSCED={'numHour': numHour,
             'numResource':numResource,
             'timeStamp':loadTimeStamp,
             'resourceName':resourceNameUnique,
			 'numYear':numYear,
             'GenType':resourceTypeUnique,
             'Load':hrLoad,
             }
    
    output = open(dataPath+'/parSCED.pkl', 'wb')
    pickle.dump(parSCED, output)
    output.close()

    #-----------------------------------------------------------------------
    j1 = 0

    hrMCP=[]
    hrMCPresource=[]
    hrMCPPair={}
    hrMWaward={}
    # loop over each hour
    for j,i in enumerate(loadTimeStamp):
        indexes = [i0 for i0,x in enumerate(scedTimeStamp[j1:]) if x == i]

        resourceName = resourceNameList[j1:][indexes]
        resourceType=resourceTypeList[j1:][indexes]
        bidPairList = bidPair[j1:][indexes]

        j1 += len(indexes)

        tempHrBidPair = {}
        tempHrMWaward={}
        startIdx = 0
        startIncIdx=0
        selectMCPresource = []
        MCPMWaward=[]
        resourceBidStartIdx=[]
        resourceBidEndIdx=[]
        resourceIncBidStartIdx = []
        resourceIncBidEndIdx = []
        bidResource = []
        bidIncResource = []
        bidMW = []
        bidPrice = []
        bidIncMW = []
        bidIncPrice = []
        #loop over each resource
        for j2 in np.arange(len(indexes)):
            # loop over each bid pair
            bidIdx=0
            incIdx = 0
            tempResourceBidPair={}

            for j3 in np.arange(len(bidPairList[j2,:])):


                if j3 %2 ==0:
                    incMW = bidPairList[j2][j3 + 2] - bidPairList[j2][j3]

                    bidResource.append(resourceName[j2])
                    bidMW.append(bidPairList[j2][j3]+0.0)
                    bidPrice.append(bidPairList[j2][j3+1]+0.0)

                    if bidIdx>0 and bidPairList[j2][j3]>0:
                        incMWtemp=bidPairList[j2][j3] - bidPairList[j2][j3-2]
                        if incMWtemp>0:
                            bidIncMW.append(incMWtemp)
                            bidIncPrice.append(bidPairList[j2][j3+1])
                            bidIncResource.append(resourceName[j2])
                            incIdx+=1
                        elif incMWtemp==0 and incMW>0:
                            bidIncPrice[startIncIdx+incIdx-1]=bidPairList[j2][j3+1]

                    bidIdx+=1

                    if incMW<0:
                        break;

            endIdx=bidIdx+startIdx
            endIncIdx=incIdx+startIncIdx

            # bid pair for each hour with each resources
            tempResourceBidPair['resourceName'] =resourceName[j2]
            tempResourceBidPair['resourceType']=resourceType[j2]
            tempResourceBidPair['bidMW']=bidMW[startIdx:endIdx]
            tempResourceBidPair['bidPrice']=bidPrice[startIdx:endIdx]
            tempResourceBidPair['incrementalMW']=bidIncMW[startIncIdx:endIncIdx]
            tempResourceBidPair['incrementalPrice']=bidIncPrice[startIncIdx:endIncIdx]

            resourceBidStartIdx.append(startIdx)
            resourceBidEndIdx.append(endIdx)
            resourceIncBidStartIdx.append(startIncIdx)
            resourceIncBidEndIdx.append(endIncIdx)

            plot=0
            #supply curve for each resource each hour
            if plot>0:
                line_up, = plt.step(np.asarray(bidMW[startIdx:endIdx-1]), np.asarray(bidPrice[startIdx:endIdx-1]), 'r-o', linewidth=2.0, label=resourceName[j2])
                plt.xlabel("MW")
                plt.ylabel('bidding price($/MWh)')
                plt.grid(True)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)

                strname=resourceName[j2]+'_'+resourceType[j2]+i.strftime('%Y-%m-%d %H-%M-%S')+'.png'
                figFile = ospath.join(currentFilePath, 'Fig/',strname)
                plt.savefig(figFile, bbox_inches='tight')
                plt.show()
                plt.close()

            startIdx = endIdx
            startIncIdx = endIncIdx



        #sort based on incremental bidding price
        temp_index, HrIncPrice = zip(*sorted([(i1, e) for i1, e in enumerate(bidIncPrice)], key=itemgetter(1)))
        # bid pair for each hour with all resources
        tempHrBidPair['resourceName'] = bidResource
        tempHrBidPair['bidMW'] = bidMW
        tempHrBidPair['bidPrice'] = bidPrice

        bidIncResource1=np.asarray(bidIncResource)
        bidIncMW1=np.asarray(bidIncMW)
        bidIncPrice1=np.asarray(bidIncPrice)
        bidMW1 = np.asarray(bidMW)
        bidPrice1 = np.asarray(bidPrice)
        temp_index=list(temp_index)
        tempHrBidPair['incResourceName'] = bidIncResource1[temp_index]
        tempHrBidPair['incrementalMW'] = bidIncMW1[temp_index]
        tempHrBidPair['incrementalPrice'] = bidIncPrice1[temp_index]

        MWPoint=[]

        tempbidMW=bidIncMW1[temp_index]
        tempIncPrice=bidIncPrice1[temp_index]
        tempResource=bidIncResource1[temp_index]
        jj = 0
        for ii in np.arange(len(temp_index)):
            if ii==0:
                MWPoint.append(tempbidMW[ii])
            else:
                MWPoint.append(MWPoint[ii-1]+tempbidMW[ii])


            if hrLoad[j]<=MWPoint[ii] and jj==0:
                #find the market clearing price
                hrMCP.append(tempIncPrice[ii])
                hrMCPresource.append(tempResource[ii])
                MCPIdx=ii+1
                jj+=1

        tempHrBidPair['MWPoint']=MWPoint

        # find winning resources
        used = set()
        selectResourcesArr = [(x.encode('UTF8')) for x in tempResource[:MCPIdx] if x not in used and (used.add(x) or True)]
        selectResources = np.asarray(selectResourcesArr)
        bidStartArr = np.asarray(resourceBidStartIdx)
        bidEndArr = np.asarray(resourceBidEndIdx)
        MWPointArr = np.asarray(MWPoint)

        # time stamp
        print i
        print "resource, award, max cap"
        # find the total MW from each resource for each hour
        for k1 in np.arange(len(selectResources)):
            idxK= [i0 for i0,x in enumerate(tempResource[:MCPIdx]) if x == selectResources[k1]]

            tempMW=tempbidMW[idxK]

            if idxK[-1]==MCPIdx-1:
                tempMW[-1]=hrLoad[j]-MWPointArr[MCPIdx-2]

            selectMCPresource.append( selectResources[k1])
            MCPMWaward.append(sum(tempMW))

            idxL = [i0 for i0, x in enumerate(resourceName) if resourceName[i0] == selectResources[k1]]

            maxcap=bidMW1[list(bidEndArr[idxL]-1)[0]]
            print selectResources[k1],sum(tempMW),maxcap
            plotAward=0

            if plotAward>0:
                t01=list(bidStartArr[idxL])
                t02=list(bidEndArr[idxL]-1)
                x1=bidMW1[t01[0]:t02[0]]
                x2=bidPrice1[t01[0]:t02[0]]
                line_up, = plt.step(np.asarray(x1), np.asarray(x2), 'r-o', linewidth=2.0, label=selectResources[k1]+'_'+str(maxcap)+'MW')
                line_down,=plt.plot(np.asarray([sum(tempMW),sum(tempMW)]),np.asarray([bidPrice1[t01[0]],bidPrice1[t02[0]-1]]),'b--',label='award'+'_'+str(sum(tempMW))+'MW')
                plt.xlabel("MW")
                plt.ylabel('bidding price($/MWh)')
                plt.grid(True)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)

                strname=i.strftime('%Y-%m-%d %H-%M-%S')+'_'+resourceType[idxL[0]]+'_'+selectResources[k1]+'_'+repr(maxcap)+'MW'+'_'+'award'+'_'+repr(sum(tempMW))+'MW'+'.png'
                figFile = ospath.join(currentFilePath, 'Fig/',strname)
                plt.savefig(figFile, bbox_inches='tight')
                #plt.show()
                plt.close()

        tempHrMWaward['resource']=selectMCPresource
        tempHrMWaward['MW']=MCPMWaward
        print 'hour, supply, load'
        print i,sum(MCPMWaward),hrLoad[j]
        #draw the bid points which meet the load
        line_up, = plt.step(np.asarray(MWPointArr[:MCPIdx]), np.asarray(tempIncPrice[:MCPIdx]), 'r-o',
                            linewidth=2.0, label=i.strftime('%Y-%m-%d %H-%M-%S'))
        line_down,=plt.plot(np.asarray([hrLoad[j],hrLoad[j]]),np.asarray([min(tempIncPrice[:MCPIdx]),max(tempIncPrice[:MCPIdx])]),'b--',label='load')
        plt.xlabel("MW")
        plt.ylabel('incremental bidding price($/MWh)')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        strname = 'load'+'_'+ i.strftime('%Y-%m-%d %H-%M-%S') + '.png'
        figFile = ospath.join(currentFilePath, 'Fig/', strname)
        plt.savefig(figFile, bbox_inches='tight')
        #plt.show()
        plt.close()

        # draw all the bid points, intersection between load and bid curve
        line_up, = plt.step(np.asarray(MWPointArr), np.asarray(tempIncPrice), 'r-o',
                            linewidth=2.0, label=i.strftime('%Y-%m-%d %H-%M-%S'))
        line_down,=plt.plot(np.asarray([hrLoad[j],hrLoad[j]]),np.asarray([min(tempIncPrice),max(tempIncPrice)]),'b--',label='load')
        plt.xlabel("MW")
        plt.ylabel('incremental bidding price($/MWh)')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        strname =  i.strftime('%Y-%m-%d %H-%M-%S') + '.png'
        figFile = ospath.join(currentFilePath, 'Fig/', strname)
        plt.savefig(figFile, bbox_inches='tight')
        #plt.show()
        plt.close()

        hrMWaward[j] = tempHrMWaward

    hrMCPPair['marginalResouce']=hrMCPresource
    hrMCPPair['MCP']=hrMCP
    hrMCPPair['loadMW']=hrLoad
    
    line_up, = plt.plot(np.arange(1,24+1),np.asarray(hrMCP),'b-x',label='MCP')
    plt.xlabel("MW")
    plt.ylabel('market clearing price($/MWh)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    strname = 'daily MCP' + '.png'
    figFile = ospath.join(currentFilePath, 'Fig/', strname)
    plt.savefig(figFile, bbox_inches='tight')
    #plt.show()
    plt.close()

if __name__ == '__main__':
    main()