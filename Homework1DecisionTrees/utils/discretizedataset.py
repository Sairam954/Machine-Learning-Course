import  pandas as pd

class DiscretizeDataset:
    def __init__(self):
        pass

    def equaldistantbin(self,datasetframe,column_name,num_of_bins):
        discretizedataset = None
        maxvalue = datasetframe[column_name].max()
        minvalue = datasetframe[column_name].min()
        bin_width = (maxvalue-minvalue)/num_of_bins
        # value = datasetframe['x1'][0]
        # print(value)

        valuebindict = {}
        for value in datasetframe[column_name]:
            minboundary = minvalue
            maxboundary = minvalue + bin_width
            # print("Value"+str(value))
            for bin in range(num_of_bins):
                # print("Bin"+str(bin))
                # print("Range "+ str(minboundary)+'-'+str(maxboundary))
                if value>=minboundary and value < maxboundary:
                    valuebindict[value] = 'bin'+str(bin)
                minboundary = maxboundary
                maxboundary = maxboundary+bin_width
            if(value==maxvalue):
                valuebindict[value]= 'bin'+str(num_of_bins-1)
            # print(value)
        datasetframe[column_name].replace(valuebindict, inplace=True)
        # print(datasetframe)
        discretizedataset = datasetframe
        return discretizedataset

# ddobj =  DiscretizeDataset()
# dataset = pd.read_csv('../input/synthetic-1.csv')
# print(ddobj.equaldistantbin(dataset,'x1',4))
