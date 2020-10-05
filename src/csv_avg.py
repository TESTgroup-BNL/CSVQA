import sys
import os
import logging
from datetime import datetime
from glob import glob
import pandas as pd

#File/directory to process
#   Can be overridden with command line parameter
#   Default is all CSV files in current directory
#   Accepts standard wildcards
fpath = "*.csv"

#Interval of exisitng data
baseFreq = '1Min'

#Intervals to group by
#   An output file will be generated for each interval
freqs = ('30min', '1H', '1D')

#Row containing column names
headerRow = 1

#Other header rows (will not be processed)
skipRows = [2,3]

#Column used as date/time index
indexCol = "TIMESTAMP"

#Columns that should be min-ed instead of averaged in grouping aggregation
timeCols = ('TIMESTAMP', 'RECORD', 'YYYYMMDD', 'HHMMSS', 'Frac_of_Day')

#Columns that should be treated as integers
intCols = ('RECORD', 'YYYYMMDD', 'HHMMSS')

#Columns that should be unique (used to identify duplicates)
uniqueCols = ["TIMESTAMP","RECORD"]

#Preference for handling duplicate records (all columns):
#   ‘first’ : Drop duplicates except for the first occurrence.
#   ‘last’ : Drop duplicates except for the last occurrence.
#   False : Drop all duplicates.
#   ‘halt’ : Stop proccessing data.
dupRecPref = 'last'

#Preference for handling duplicate timestamps:
#   ‘first’ : Drop duplicates except for the first occurrence.
#   ‘last’ : Drop duplicates except for the last occurrence.
#   False : Drop all duplicates.
#   ‘halt’ : Stop proccessing data.
dupTSPref = 'last'

#Maximum fraction of records that may be missing from a group mean (0.1 = 10%)
maxMissing = 0.1

#Value to read as Nans from the input files
inputNans = ["NAN","-999"]

#Value to use for Nans in the output files
outputNans = "Nan"

#Output precision
#   'original' : use original column precision (maximum from original data)
#   else provide format string (ex: '%0.2f')
outputDecs = 'original'

#Minimum/Maximum output precision
#   Useful when using 'original' above or with repeating decimals in floating point values to include or exclude decimal places to something reasonable
outputDecsMin = 2
outputDecsMax = 6

#Console log level
#   'INFO' : Shows processing steps only
#   'DEBUG' : Shows details of actual data (column summary, duplicate records, missing data stats)
consoleLogLevel = logging.INFO



def setupLogging(fpath):
    global logCurrentFile
    logCurrentFile = ""

    formatter = logging.Formatter('%(asctime)s\t[%(levelname)s]:\t%(logCurrentFile)s%(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(consoleLogLevel)

    logFile = os.path.join(os.path.dirname(os.path.abspath(fpath)), "csv_avg_"+ datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log")
    fh = logging.FileHandler(logFile, mode='w')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.DEBUG,format=formatter, handlers=[fh, ch])
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        global logCurrentFile
        record = old_factory(*args, **kwargs)
        record.logCurrentFile = logCurrentFile
        return record

    logging.setLogRecordFactory(record_factory)

def logConfig():
    config = {"fpath":fpath,"baseFreq": baseFreq,"freqs": freqs,"headerRow": headerRow, "skipRows": skipRows, "indexCol": indexCol, "timeCols": timeCols,"intCols": intCols, \
              "uniqueCols": uniqueCols, "dupRecPref": dupRecPref, "dupTSPref": dupTSPref,"maxMissing": maxMissing,"inputNans": inputNans, "outputNans": outputNans, \
              "outputDecs": outputDecs,"outputDecsMin": outputDecsMin, "outputDecsMax": outputDecsMax,"consoleLogLevel": consoleLogLevel}
    
    configStr = ""
    for k, i in config.items():
        configStr += "\n\t" + k + ": " + str(i)

    logging.debug("Configuration:" + configStr)

def getFiles(fpath):
    return [f for f in glob(fpath)]


def loadData(fname, ts=indexCol, nan_vals=inputNans):
    global logCurrentFile, maxDec

    logging.info("Loading " + fname + "...")
    logCurrentFile = "[" + fname + "]\t"
    d = pd.read_csv(fname, header=[headerRow], skiprows=skipRows, parse_dates=[ts], na_values=nan_vals)
    d.index = pd.DatetimeIndex(d[ts])
    logging.debug(str(d.info()).replace("\n","\n\t"))

    logging.info("Calculating column precisions...")

    def getDecimalPlaces(x):
        x_split = x.split('.')
        if len(x_split) > 1:
            return len(x.split('.')[1])
        else:
            return 0

    maxDec = {}
    for col in d.columns:
        if col not in timeCols:
            origDecs = max(d[col].astype('str').apply(getDecimalPlaces))
            minBound = max(origDecs, outputDecsMin)
            maxBound = min(minBound, outputDecsMax)
            maxDec[col] = maxBound
    logging.debug("\n\t" + str(maxDec).replace(",","\n\t"))

    return d


def checkDups(df, dupPref='last', cols=None):
    if cols==None:
        logging.info("Checking for duplicates in any column...")
    else:
        logging.info("Checking for duplicates in "+str(cols)+"...")
    dups = d.duplicated(subset=cols, keep=False)

    if sum(dups)==0:
        logging.info("No duplicates found.")
        return df
    else:
        logging.info("Found " + str(sum(dups)) + " duplicates")
        logging.debug("\n" + str(d[dups]).replace("\n","\n\t\t"))

        if dupPref=='halt':
            logging.info("\tStopped processing due to dupRecPref = 'halt'")
            sys.exit()
        else:
            df.drop_duplicates(subset=cols, keep=dupPref, inplace=True)
            logging.info ("\tDropped duplicates, kept " + dupPref + ".")
            return df


def checkOrder(df, ts=indexCol):
    logging.info("Checking record order...")
    if d[ts].is_monotonic:
        logging.info("\tNo out of order timestamps found.")
    else:
        logging.info("\tRecords are out of order, cannot continue.")
        sys.exit()


def fillMissing(df, freq=baseFreq):
    logging.info("Checking for missings rows...")
    old_len = len(df)
    new_idx =  pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df_reindexed = df.reindex(new_idx)
    logging.info("\tFilled "+str(len(df_reindexed)-old_len)+" missing rows")
    return df_reindexed


def avgToFreqs(df, ts=indexCol):
    logging.info("Resampling data...")
    agg_funcs = {}
    for col in d.columns:
        if col == ts:
            continue
        elif col in timeCols:
            agg_funcs.update({col: 'min'})
        else:
            agg_funcs.update({col: 'mean'})

    d_out = {}
    for f in freqs:
        logging.info("\tInterval: " + f)
        d_resampled = d.resample(f)
        d_resampled_size = d_resampled.apply(lambda x: x.isnull().sum()/len(x))             #Calculates fraction of nan values in each group

        minCountNotReached = d_resampled_size.iloc[:,1:] > maxMissing                       #Creates array where True indicates less than the required values are present in a group
        
        d_out[f] = d_resampled.agg(agg_funcs)
        d_out[f].mask(minCountNotReached, inplace=True)

        missingData = round(d_out[f].isnull().sum()/len(d_out[f])*100, 2)
        logging.debug("\n\t\t" + str(d_out[f]).replace("\n","\n\t\t"))
        logging.debug("\n\tData missing per column (%):\n\t\t" + str(missingData).replace("\n","\n\t\t"))

    return d_out

def applyDataTypes(df):
    #Fix integer columns
    for intCol in intCols:
        df[intCol] = df[intCol].map(lambda x: '%i' % x if not pd.isna(x) else outputNans)

    #Apply precision
    if outputDecs == 'original':
        for col in df.columns:
            if (col not in timeCols) and (col not in intCols):
                df[col] = df[col].map(lambda x: ('%0.' + str(maxDec[col]) + 'f') % x if not pd.isna(x) else outputNans)
    #else default will be applied when writing output


def genOutputFname(fname, freq):
    orig = os.path.splitext(fname)
    return orig[0] + "_avg_" + freq + orig[1]


def saveOutput(d, freq):
    if outputDecs == 'original':
        d.to_csv(genOutputFname(fname,freq), na_rep=outputNans, index_label=indexCol)
    else:
        d.to_csv(genOutputFname(fname,freq), na_rep=outputNans, float_format=outputDecs, index_label=indexCol)



if __name__ == "__main__":
    try:
        fpath = sys.argv[1]
    except Exception:
        logging.warning("No file path provded, using default ('./*.csv')")

    setupLogging(fpath)

    logConfig()

    for fname in getFiles(fpath):
        
        #Load Data
        d = loadData(fname)

        #Remove duplicate records
        checkDups(d, dupPref=dupRecPref, cols=uniqueCols)

        #Remove any remaining records with duplicate timestamps
        checkDups(d, dupPref=dupTSPref, cols=indexCol)

        #Check that records are in order
        checkOrder(d)

        #Requires 'd =' because reindexing cannot be done in place if rows are added
        d = fillMissing(d)

        #Resample data into intervals specified by 'freqs'
        d_out = avgToFreqs(d)

        for d_freq in d_out:
            #Apply data types and precision
            applyDataTypes(d_out[d_freq])

            #Save data
            saveOutput(d_out[d_freq], d_freq)

    logCurrentFile = ""
    logging.info("Done.")