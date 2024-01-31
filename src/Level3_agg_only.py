import sys
import os
import logging
from datetime import datetime
from glob import glob
import pandas as pd
import numpy as np
import configparser
from ast import literal_eval as le
from datetime import datetime


def readConfig():
    #Note: using print here instead of logging because logger isn't initialized
    try:
        fpath = None
        if sys.argv[1][-4:] == "conf":
            configFile = sys.argv[1]
            print("Using config file: " + configFile)
        else:
            fpath = sys.argv[1]
            configFile = "default.conf"
            print("Using source path: " + fpath)
    except Exception:
        
        print("No file paths provded, using defaults ('default.conf', './*.csv')")
        fpath = None
        configFile = "default.conf"

    try:
        cp = configparser.RawConfigParser()
        cp.optionxform = str    #changes the default parser to be case-sensitive
        cp.read(configFile)
    except Exception as e:
        print("Error loading config file: " + str(e))
        exit(e)     

    opts = {}
    try:    
        for s in cp.sections():
            opts[s] = {}
            for o in cp[s]:
                try:
                    opts[s][o] = le(cp[s][o])
                except Exception:
                    opts[s][o] = cp[s][o]

        if fpath is not None:
            opts["Input"]["fpath"] = fpath
            #opts["Input"]["fpath"] = opts["Input"]["fpath"] #.replace("\\","\\\\")

        if not os.path.isabs(opts["Input"]["sourceCharsFile"]):
            opts["Input"]["sourceCharsFile"] = os.path.join(opts["Input"]["fpath"], opts["Input"]["sourceCharsFile"])

        if opts["Output"]["outputPath"] == "":
            opts["Output"]["outputPath"] = opts["Input"]["fpath"]

        if opts["Output"]["consoleLogLevel"] in ("NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            opts["Output"]["consoleLogLevel"] = eval("logging." + opts["Output"]["consoleLogLevel"])

    except Exception as e:
        print("Error processing " + configFile + ": " + str(e))
        exit(e)

    setupLogging(opts["Output"]["outputPath"], consoleLogLevel=opts["Output"]["consoleLogLevel"])
    logging.debug("Configuration:" + str(opts))

    return opts


def setupLogging(fpath, consoleLogLevel=0):
    global logCurrentFile, logging
    logCurrentFile = ""

    format_str = '%(asctime)s\t[%(levelname)s]:\t%(logCurrentFile)s%(message)s'
    formatter = logging.Formatter(format_str)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(consoleLogLevel)

    logFile = os.path.join(os.path.abspath(fpath), "Level2to3_"+ datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log")
    fh = logging.FileHandler(logFile, mode='w')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    
    logging.basicConfig(level=logging.DEBUG, format=format_str, handlers=[fh, ch])
    logging.root.setLevel(logging.NOTSET)
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        global logCurrentFile
        record = old_factory(*args, **kwargs)
        record.logCurrentFile = logCurrentFile
        return record

    logging.setLogRecordFactory(record_factory)
    

def getFiles(fpath, altFileSuffix=None):
    f_list = [f for f in glob(fpath)]
    if not altFileSuffix is None:
        f_list_filtered = [f for f in f_list if os.path.splitext(f)[0][-len(altFileSuffix):] != altFileSuffix]
    return f_list_filtered


def loadData(fname, ts="TIMESTAMP", nan_vals=["NAN","-999"], headerRow=0, skipRows=[], dtFormat=None, *args, **kwargs):
    global logCurrentFile

    logging.info("Loading " + fname + "...")
    logCurrentFile = "[" + os.path.split(fname)[1] + "]\t"
    try:
        if dtFormat is None:
            d = pd.read_csv(fname, header=[headerRow], skiprows=skipRows, parse_dates=[ts], na_values=nan_vals)
        else:
            dateparse = lambda x_d: datetime.strptime(x_d, dtFormat)   
            d = pd.read_csv(fname, header=[headerRow], skiprows=skipRows, parse_dates=[ts], date_parser=dateparse, na_values=nan_vals)
        d.index = pd.DatetimeIndex(d[ts]).rename("idx")
        logging.debug(str(d.info()).replace("\n","\n\t"))
        return d
    except Exception as e:
        logging.warning("Problem while loading " + os.path.split(fname)[1] + ": " + str(e))

    return None


def loadSourceChars(sourceCharsFile):
    d_chars = pd.read_csv(sourceCharsFile, header=[0])
    
    sourceChars = {}
    for x in d_chars.iloc:
        try:
            if x[0]=="Start":
                sourceChars["start"] = datetime.strptime(x[1].strip(), "%Y-%m-%d %H:%M:%S")
            if x[0]=="End":
                sourceChars["end"] = datetime.strptime(x[1].strip(), "%Y-%m-%d %H:%M:%S")

            for c in x["Columns"].split(';'):
                sourceChars[c] = {}
                try:
                    sourceChars[c]["fixedDecPlaces"] = int(x["Useful Decimal Places"])
                except Exception:
                    sourceChars[c]["fixedDecPlaces"] = None
                
                try:
                    sourceChars[c]["minVal"] = float(x["Min"])
                except Exception:
                    sourceChars[c]["minVal"] = None

                try:
                    sourceChars[c]["maxVal"] = float(x["Max"])
                except Exception:
                    sourceChars[c]["maxVal"] = None
        except Exception:
            pass

    return sourceChars


def calcDecPlaces(d, timeCols, outputDecs, outputDecsMin, outputDecsMax, sourceChars):

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
            try:
                try:
                    #Use defined precision
                    maxDec[col] = sourceChars[col]["fixedDecPlaces"]
                    if sourceChars[col]["fixedDecPlaces"]==None:
                        raise KeyError
                except KeyError:
                    #Fallback on general rules
                    origDecs = max(d[col].astype('str').apply(getDecimalPlaces))
                    if outputDecs == "original":
                        minBound = max(origDecs, outputDecsMin)
                        maxBound = min(minBound, outputDecsMax)
                        maxDec[col] = maxBound
                    else:
                        maxDec[col] = outputDecs
                    
            except Exception as e:
                logging.error("Error processing " + col + ": " + str(e))

    logging.debug("\n\t" + str(maxDec).replace(",","\n\t"))

    return maxDec


def calcQAStats(d, d_qa, d_dev):
    #QA Mask Stats
    try:
        d_stats = d_qa.drop("idx", axis=1).apply(pd.value_counts)
    except KeyError:
        d_stats = d_qa.apply(pd.value_counts)
    d_stats = d_stats.reindex(range(7))
    d_stats = d_stats.mask(d_stats.isna(),0).astype(int)
    
    #Alt source correlations
    if d_dev is not None:
        s_coor = [pd.Series([],dtype=float,name="corr " + n) for n in ("raw","0","1","2","3","4","5","6","7","8","9")]
        d_stats = d_stats.append(s_coor)

        for c in d_stats.columns:
            try:
                d_stats[c]["corr raw"] = d_dev[c].corr(d_dev[c + "_alt"])
                for n in range(7):
                    d_stats[c]["corr " + str(n)] = d_dev[c][d_qa[c]==n].corr(d_dev[c + "_alt"][d_qa[c]==n])
                    
            except KeyError:
                pass
            except Exception as e:
                logging.warning("Error calculating stats: " + str(e))

    #Raw data stats
    d_stats = d_stats.append(d.describe())

    return d_stats
    

def avgToFreqs(d, ts="TIMESTAMP", maxMissing=0.1, timeCols=["TIMESTAMP"], freqs=[], excludeMissing=[], *args, **kwargs):
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
        
        try:
            for col in excludeMissing:
                minCountNotReached[col] = False                                             #Unmarks columns where Nans should be ignored
        except KeyError:
            pass

        d_out[f] = d_resampled.agg(agg_funcs)
        d_out[f].mask(minCountNotReached, inplace=True)

        missingData = round(d_out[f].isnull().sum()/len(d_out[f])*100, 2)
        logging.debug("\n\t\t" + str(d_out[f]).replace("\n","\n\t\t"))
        logging.debug("\n\tData missing per column (%):\n\t\t" + str(missingData).replace("\n","\n\t\t"))

    return d_out


def applyDataTypes(df, intCols, timeCols, outputNans, outputDecs, colDecs):
    #Fix integer columns
    for intCol in intCols:
        try:
            df[intCol] = df[intCol].map(lambda x: '%i' % x if not pd.isna(x) else outputNans)
        except KeyError:
            logging.warning("Column defined in config but not present: " + intCol)

    #Apply precision
    if outputDecs == 'original':
        for col in df.columns:
            if (col not in timeCols) and (col not in intCols):
                df[col] = df[col].map(lambda x: ('%0.' + str(colDecs[col]) + 'f') % x if not pd.isna(x) else outputNans)
    #else default will be applied when writing output


def genOutputFname(fpath, fname, sub):
    orig = os.path.splitext(os.path.join(fpath, os.path.split(fname)[1]))
    return orig[0] + sub + ".csv"


def saveOutput(d, fpath, fname, sub, outputNans, indexCol=None, index=True):
    #if outputDecs == 'original':
    d.to_csv(genOutputFname(fpath,fname,sub), na_rep=outputNans, index_label=indexCol, index=index)
    #else:
    #    d.to_csv(genOutputFname(fname,sub), na_rep=outputNans, float_format=outputDecs, index_label=indexCol)


def applyQA(d, d_qa, d_dev):
    logging.info("Applying QA mask...")
      
    #Replace out of range values with Nans
    d.mask(d_qa==2, np.nan, inplace=True)

    #Fill alternate data
    for c in d:
        try:
            d[c].mask(d_qa[c].isin((4,6)), d_dev[c + "_alt"], inplace=True)
        except KeyError:
            pass
        except Exception as e:
            logging.warning("Error applying alternate data for " + c + ": " + str(e))

    #Trim start and end times
    d = d[~d_qa["TIMESTAMP"].isin((1,3,5,7,9))]

    return d


if __name__ == "__main__":

    logCurrentFile = ""
    opts = readConfig()

    fPath = genOutputFname(opts["Output"]["outputPath"],opts["Input"]["inputFile"],"_Level3")

    for fname in getFiles(fPath, opts["Level 1"]["altFileSuffix"]):

        fname_out = fname.replace("_Level3","")
        
        #Load Level 2 data
        d = loadData(fname, ts=opts["Input"]["indexCol"], nan_vals=opts["Output"]["outputNans"])
        if d is None:
            continue
        
        logCurrentFile = "[" + os.path.split(fname)[1] + "]\t"

        #Load source characteristics
        sourceChars = loadSourceChars(opts["Input"]["sourceCharsFile"])
        
        #Calc column precisions
        colDecs = calcDecPlaces(d, opts["Input"]["intCols"], opts["Output"]["outputDecs"], opts["Output"]["outputDecsMin"], opts["Output"]["outputDecsMax"], sourceChars)

        #Resample data into intervals specified by 'freqs'
        #Note: this is done before saving the base freq so that resampling is done before apply data types to base data
        d_out = avgToFreqs(d, timeCols=opts["Input"]["timeCols"], **opts["Level 3"])

        #Save base freq
        applyDataTypes(d, opts["Input"]["intCols"], opts["Input"]["timeCols"], opts["Output"]["outputNans"], opts["Output"]["outputDecs"], colDecs)
        saveOutput(d, opts["Output"]["outputPath"], fname_out, "_Level3_agg", outputNans=opts["Output"]["outputNans"], index=False)

        #Save additional sample intervals
        for d_freq in d_out:
            #Apply data types and precision
            applyDataTypes(d_out[d_freq], opts["Input"]["intCols"], opts["Input"]["timeCols"], opts["Output"]["outputNans"], opts["Output"]["outputDecs"], colDecs)

            #Save Level 3 data
            saveOutput(d_out[d_freq], opts["Output"]["outputPath"], fname_out, "_Level3_agg_" + d_freq, outputNans=opts["Output"]["outputNans"], indexCol=opts["Input"]["indexCol"])

    logCurrentFile = ""
    logging.info("Done.")