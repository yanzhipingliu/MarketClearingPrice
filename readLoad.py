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

def ensure_datetime(d):
    """
    Takes a date or a datetime as input, outputs a datetime
    """
    if isinstance(d, datetime):
        return d
    return datetime(d.year, d.month, d.day,d.hour,d.minute,d.second)

def datetime_cmp(d1, d2):
    """
    Compares two timestamps.  Tolerates dates.
    """
    return cmp(ensure_datetime(d1), ensure_datetime(d2))

def print_library_info(cnxn):
    # import pyodbc
    print 'python:  %s' % sys.version
    print 'pyodbc:  %s %s' % (pyodbc.version, ospath.abspath(pyodbc.__file__))
    print 'odbc:    %s' % cnxn.getinfo(pyodbc.SQL_ODBC_VER)
    print 'driver:  %s %s' % (cnxn.getinfo(pyodbc.SQL_DRIVER_NAME), cnxn.getinfo(pyodbc.SQL_DRIVER_VER))
    print '         supports ODBC version %s' % cnxn.getinfo(pyodbc.SQL_DRIVER_ODBC_VER)
    print 'os:      %s' % platform.system()
    print 'unicode: Py_Unicode=%s SQLWCHAR=%s' % (pyodbc.UNICODE_SIZE, pyodbc.SQLWCHAR_SIZE)

    if platform.system() == 'Windows':
        print '         %s' % ' '.join([s for s in platform.win32_ver() if s])

def main():
    # currentFile1 = ospath.dirname(inspect.getfile(inspect.currentframe()))
    '''
    pathName = ospath.join(currentFile,'Outputs/')
    mpsOutFile = pathName+'/MyMPS.mps'
    '''
    # currentFile = 'Z:/software/Python/example/access/accessexam1.py'
    # path = ospath.dirname(ospath.abspath(currentFile))


    #databaseFile = ospath.join(path, 'Data/JHU_WECC300_Load.accdb')

    currentFilePath = ospath.dirname(inspect.getfile(inspect.currentframe()))
    # pathName="12Mon_2day_24hr//"
    path = currentFilePath
    # write the MPS file???
    # mpsOutFile=pathName+'/Model%s.mps'%time.ctime().translate(None,': ')
    loadFile = path + '/DemandData.csv'


    scedFile = ospath.join(ospath.dirname(__file__), 'SCED_Data.csv')

    assert ospath.exists(loadFile)

    tempFile = ospath.join(path,'Fig')
    if not ospath.exists(tempFile):
        os.makedirs(tempFile)
    
    tempFile = ospath.join(path,'Output')
    if not ospath.exists(tempFile):
        os.makedirs(tempFile)
    dataPath = ospath.join(path, 'Output')

    print("read load file")
    df = pandas.read_csv(loadFile);
    columnnames = list(df.columns.values)
    print columnnames
    tempLoad = df.values
    loadTimeStamp1 = tempLoad[:, 0]

    #loadTimeStamp = [int(x) for x in loadTimeStamp]

    #TEPPCAreaList = columnnames[1:]
    hrLoad = tempLoad[:, 1]
    # https://stackoverflow.com/questions/4539254/how-to-get-datatypes-of-specific-fields-of-an-access-database-using-pyodbc
    print("read SCED file")

    df = pandas.read_csv(scedFile);
    #df = pandas.read_csv('U:\Documents\Code\Python_LoadReshape\V3\Data/2024TEPPCAreaLoad.csv')
    # print df
    columnnames = list(df.columns.values)
    print columnnames



    tempSCED = df.values
    scedTimeStamp1=tempSCED[:,0]
    # before sorted
    t1 = datetime.strptime(scedTimeStamp1[0], "%Y-%m-%d %H:%M:%S")
    t2 = datetime.strptime(loadTimeStamp1[0], "%Y-%m-%d %H:%M:%S")
    difference = t1 - t2
    print(difference.seconds)
    #after sorted
    #loadTimeStamp = pandas.to_datetime(loadTimeStamp1, format="%Y-%m-%d %H:%M:%S").sort_values()
    #scedTimeStamp =    pandas.to_datetime(scedTimeStamp1, format="%Y-%m-%d %H:%M:%S").sort_values()

    #timeStamp = [int(x) for x in timeStamp]
    #storedTime = datetime.strptime(scedTimeStamp[0], "%Y-%m-%d %H:%M:%S.%f")
    #t0=datetime_cmp(scedTimeStamp[0], loadTimeStamp[0])

    #scedTimeStamp=sorted([datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in scedTimeStamp1])
    loadTimeStamp = sorted([datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in loadTimeStamp1])
    #sorted_items, sorted_inds = zip(*sorted([(i,e) for i,e in enumerate(my_list)], key=itemgetter(1)))
    sorted_inds,scedTimeStamp = zip(*sorted([(e,datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")) for e,dt in enumerate(scedTimeStamp1)], key=itemgetter(1)))

    t1 = scedTimeStamp[0]
    t2 = loadTimeStamp[0]
    #t1=scedTimeStamp[0].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    #t2=loadTimeStamp1[0].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    difference = t1 - t2

    print(difference.seconds)  # 380, in this case

    latest = max((t1, t2))

    #TEPPCAreaList = columnnames[1:]
    used = set()
    resourceNameList=tempSCED[sorted_inds, 1]
    #x     for x in mylist if x not in used
    resourceNameArr = [(x.encode('UTF8')) for x in resourceNameList if x not in used and (used.add(x) or True)]
    resourceNameUnique = np.asarray(resourceNameArr)

    used = set()
    resourceTypeList = tempSCED[sorted_inds, 2]
    resourceTypeArr = [(x.encode('UTF8')) for x in resourceTypeList if x not in used and (used.add(x) or True)]
    resourceTypeUnique = np.asarray(resourceTypeArr)

    output = open(dataPath+'/resourceNameUnique.pkl', 'wb')
    pickle.dump(resourceNameUnique, output)
    output.close()

    sio.savemat(dataPath+'/resourceNameUnique.mat', {'resourceNameUnique': resourceNameUnique})
    sio.savemat(dataPath + '/resourceTypeUnique.mat', {'resourceTypeUnique': resourceTypeUnique})

    bidPair =tempSCED[sorted_inds,3:]
    tempbid=[bidPair[i,1] for i in np.arange(len(sorted_inds)) if bidPair[i,1]>-250]
    #-------------------------------end data reading--------------------------
    # cnxn.close()
    print ("\nComplete Input data reading.")

    # -------------------------reorganize load based on Area -----------------------------
    # numLoad = len(Area_Load)
    numHour = len(loadTimeStamp)#8784
    # numArea=numLoad/numHour
    numResource = len(resourceNameUnique)
    #numBus = len(UniBusID)

    # planning horizon for co-optimization
    numYear = 1#len(loadYr)#20
    parSCED={'numHour': numHour,
             'numResource':numResource,
             'timeStamp':loadTimeStamp,
             'resourceName':resourceNameUnique,
			 'numYear':numYear,
             'GenType':resourceTypeUnique,
             'Load':hrLoad,
             }
    
    #sio.savemat(dataPath + '/parSCED.mat', {'parSCED': parSCED})
    output = open(dataPath+'/parSCED.pkl', 'wb')
    pickle.dump(parSCED, output)
    output.close()

    # AreaLoad = (np.zeros((parLoad['numArea'], parLoad['numHour']), dtype=np.float32))
    #BusLoad = (np.zeros((parLoad['numBus'], parLoad['numHour']), dtype=np.float32))
    #AreaLoad = (AreaLoad.T).reshape((parLoad['numResource'], parLoad['numHour']))

    #BusLoadPeak = (np.zeros((parLoad['numBus'], 1), dtype=np.float32))
    # BusAGR = np.zeros((parLoad['numBus'], 1),dtype=np.float32)
    # BusPGR = np.zeros((parLoad['numBus'], 1), dtype=np.float32)

    # BusLoadSort_vals = (np.zeros((parLoad['numBus'], parLoad['numHour']), dtype=np.float32))
    # BusLoadSort_index = (np.zeros((parLoad['numBus'], parLoad['numHour']), dtype=np.int32))

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
        selectResources=[]
        tempbidMW=bidIncMW1[temp_index]
        tempIncPrice=bidIncPrice1[temp_index]
        tempResource=bidIncResource1[temp_index]
        jj = 0
        for ii in np.arange(len(temp_index)):
            if ii==0:
                MWPoint.append(tempbidMW[ii])
            else:
                MWPoint.append(MWPoint[ii-1]+tempbidMW[ii])
            #if jj==0:
            #    if ii==0 or (ii>0 and tempResource[ii] != tempResource[ii-1]):
            #        selectResources.append(tempResource[ii])

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
            d1=idxK[-1]
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
                line_up, = plt.step(np.asarray(x1), np.asarray(x2), 'r-o', linewidth=2.0, label=selectResources[k1])
                line_down,=plt.plot(np.asarray([sum(tempMW),sum(tempMW)]),np.asarray([bidPrice1[t01[0]],bidPrice1[t02[0]-1]]))
                plt.xlabel("MW")
                plt.ylabel('bidding price($/MWh)')
                plt.grid(True)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)

                strname=i.strftime('%Y-%m-%d %H-%M-%S')+'_'+resourceType[idxL[0]]+'_'+selectResources[k1]+'_'+repr(maxcap)+'MW'+'_'+'award'+'_'+repr(sum(tempMW))+'MW'+'.png'
                figFile = ospath.join(currentFilePath, 'Fig/',strname)
                plt.savefig(figFile, bbox_inches='tight')
                plt.show()
                plt.close()

        tempHrMWaward['resource']=selectMCPresource
        tempHrMWaward['MW']=MCPMWaward
        print sum(MCPMWaward),hrLoad[j]
        #draw the bid points which meet the load
        line_up, = plt.step(np.asarray(MWPointArr[:MCPIdx]), np.asarray(tempIncPrice[:MCPIdx]), 'b-o',
                            linewidth=2.0, label=i.strftime('%Y-%m-%d %H-%M-%S'))
        plt.xlabel("MW")
        plt.ylabel('incremental bidding price($/MWh)')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        strname = 'load'+'_'+ i.strftime('%Y-%m-%d %H-%M-%S') + '.png'
        figFile = ospath.join(currentFilePath, 'Fig/', strname)
        plt.savefig(figFile, bbox_inches='tight')
        plt.show()
        plt.close()

        # draw all the bid points, intersection between load and bid curve
        line_up, = plt.step(np.asarray(MWPointArr), np.asarray(tempIncPrice), 'b-o',
                            linewidth=2.0, label=i.strftime('%Y-%m-%d %H-%M-%S'))
        line_down,=plt.plot(np.asarray([hrLoad[j],hrLoad[j]]),np.asarray([min(tempIncPrice),max(tempIncPrice)]))
        plt.xlabel("MW")
        plt.ylabel('incremental bidding price($/MWh)')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        strname =  i.strftime('%Y-%m-%d %H-%M-%S') + '.png'
        figFile = ospath.join(currentFilePath, 'Fig/', strname)
        plt.savefig(figFile, bbox_inches='tight')
        plt.show()
        plt.close()

        hrMWaward[j] = tempHrMWaward

    hrMCPPair['marginalResouce']=hrMCPresource
    hrMCPPair['MCP']=hrMCP
    hrMCPPair['loadMW']=hrLoad

    line_up, = plt.plot(np.arange(1,24+1),hrMCP)
    plt.xlabel("MW")
    plt.ylabel('market clearing price($/MWh)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    strname = 'daily MCP' + '.png'
    figFile = ospath.join(currentFilePath, 'Fig/', strname)
    plt.savefig(figFile, bbox_inches='tight')
    plt.show()
    plt.close()

    output = open(dataPath + '/hrMCPPair.pkl', 'wb')
    cPickle.dump(hrMCPPair, output)
    output.close()
    sio.savemat(dataPath + '/hrMCPPair.mat', {'hrMCPPair': hrMCPPair})
    np.savetxt(dataPath + 'hrMCPPair.out', hrMCPPair, delimiter=',')

if __name__ == '__main__':
    main()