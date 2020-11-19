import sys
import os
import logging
from datetime import datetime
from glob import glob
import pandas as pd
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

    logFile = os.path.join(os.path.abspath(fpath), "Level0to1_"+ datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log")
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
        d.index = pd.DatetimeIndex(d[ts])
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
            if x["Sensor"]=="Start":
                sourceChars["start"] = datetime.strptime(x["Unit"].strip(), "%Y-%m-%d %H:%M:%S")
            if x["Sensor"]=="End":
                sourceChars["end"] = datetime.strptime(x["Unit"].strip(), "%Y-%m-%d %H:%M:%S")

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


def checkOrder(df, ts="TIMESTAMP"):
    logging.info("Checking record order...")
    if df[ts].is_monotonic:
        logging.info("\tNo out of order timestamps found.")
    else:
        logging.info("\tRecords are out of order, cannot continue.")
        sys.exit()


def fillMissing(df, freq="1Min", method=None, indexCol="TIMESTAMP", replaceIndexCol=True):
    logging.info("Checking for missings rows...")
    old_len = len(df)
    new_idx =  pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df_reindexed = df.reindex(new_idx, method=method)
    if replaceIndexCol:
        df_reindexed[indexCol] = df_reindexed.index
    logging.info("\tFilled "+str(len(df_reindexed)-old_len)+" missing rows")
    return df_reindexed


def calcAltDevs(d, fname, altFileSuffix, altTimeOffset, altDateTimeFormat, indexCol, inputNans, dupTSPref, baseFreq, *args, **kwargs):
    fname_head, fname_ext = os.path.splitext(fname)
    fname = fname_head + altFileSuffix + fname_ext
    #Load alt data
    d_alt = loadData(fname, ts=indexCol, nan_vals=inputNans, headerRow=0, skippedRows=[], dtFormat=altDateTimeFormat)
    #Remove any remaining records with duplicate timestamps
    checkDups(d_alt, dupPref=dupTSPref, cols=indexCol)

    #Check that records are in order
    checkOrder(d_alt)

    #Fill in missing rows
    d_alt = fillMissing(d_alt, freq=baseFreq, method='bfill', indexCol=opts["Input"]["indexCol"], replaceIndexCol=False)

    d_dev = pd.DataFrame(d[indexCol], index=d[indexCol])
    for c in d_alt.columns:
        try:
            logging.info("Checking devation for " + c)
            d_dev[c] = d[c]
            d_dev[c + "_alt"] = d_alt[c]
            try:
                d_dev[c + "_dev"] = d_alt[c] - d[c]
            except Exception as e:
                logging.error("Could not compare alt data for " + c + ": " + e)
        except KeyError:
            logging.info("\tNo matching primary column, skipping.")

    return d_dev


def createQAdf(d, sourceChars, indexCol, devThreshold=0.1, d_dev=None):
    #Create QA dataframe
    d_qa = pd.DataFrame().reindex_like(d).fillna(0)
    d_qa = d_qa.astype(int)
   
    for c in d_qa.columns:
        #Check min/max bounds
        try:
            if (c != indexCol) and not (pd.isna(sourceChars[c]["minVal"]) & pd.isna(sourceChars[c]["maxVal"]==pd.NA)):
                d_qa[c][(d[c] < sourceChars[c]["minVal"]) | (d[c] > sourceChars[c]["maxVal"])] = 2
                logging.info("For " + c + ", using min val: " + str(sourceChars[c]["minVal"]) + ", max val: " + str(sourceChars[c]["maxVal"]))
        except KeyError:
            pass
        except Exception as e:
            logging.warning("Min/max values invalid for " + c + ": " + str(e))

        if d_dev is not None:
            #Check deviation from alternate source
            try:
                if c != indexCol:
                    devCol = c + "_dev"
                    maxDev = (sourceChars[c]["maxVal"] - sourceChars[c]["minVal"])*devThreshold
                    d_qa[c] = d_qa[c].mask(abs(d_dev[devCol]) >  maxDev, d_qa[c] + 3)
                    logging.info("For " + c + ", using max deviation: " + str(maxDev))
            except KeyError:
                pass
            except Exception as e:
                logging.warning("Source characteristics or alternate data invalid for " + c + ": " + str(e))

    #Check start and end dates/times
    try:
        d_qa[(d_qa.index < sourceChars["start"]) | (d_qa.index > sourceChars["end"])] += 1
        logging.info("Start bound: " + str(sourceChars["start"]) + "\tEnd bound: " + str(sourceChars["end"]))
    except KeyError:
        pass
    except Exception as e:
        logging.warning("Start/end dates/times invalid: " + str(e))


    return d_qa


def calcQAStats(d, d_qa, d_dev):
    #QA Mask Stats
    d_stats = d_qa.apply(pd.value_counts)
    d_stats = d_stats.reindex(range(7))
    d_stats = d_stats.mask(d_stats.isna(),0).astype(int)
    
    #Alt source correlations
    if d_dev is not None:
        s_coor = [pd.Series([],dtype=float,name="corr " + n) for n in ("raw","0","1","2","3","4","5","6")]
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
    

def genOutputFname(fpath, fname, sub):
    orig = os.path.splitext(os.path.join(fpath, os.path.split(fname)[1]))
    return orig[0] + sub + ".csv"


def saveOutput(d, fpath, fname, sub, outputNans, indexCol=None, index=True):
    try:
        try:
            d.to_csv(genOutputFname(fpath,fname,sub), na_rep=outputNans, index_label=indexCol, index=index)
                
        except PermissionError:
            #If file exists and is open, append n up to 100 (so we don't get stuck forever)
            n = 0
            while(n<100):
                try:
                    d.to_csv(genOutputFname(fpath,fname,sub+str(n)), na_rep=outputNans, index_label=indexCol, index=index)
                    break
                except PermissionError:
                    n += 1
    except Exception as e:
        logging.error("Error writing file " + genOutputFname(fpath,fname,sub) + ": " + str(e))


if __name__ == "__main__":

    logCurrentFile = ""
    opts = readConfig()

    fPath = os.path.join(opts["Input"]["fpath"],opts["Input"]["inputFile"])

    for fname in getFiles(fPath, opts["Level 1"]["altFileSuffix"]):
        
        #Load data
        d = loadData(fname, **opts["Input"])
        if d is None:
            continue

        #Load source characteristics
        sourceChars = loadSourceChars(opts["Input"]["sourceCharsFile"])

        #Remove duplicate records
        checkDups(d, dupPref=opts["Level 1"]["dupRecPref"], cols=opts["Input"]["uniqueCols"])

        #Remove any remaining records with duplicate timestamps
        checkDups(d, dupPref=opts["Level 1"]["dupTSPref"], cols=opts["Input"]["indexCol"])

        #Check that records are in order
        checkOrder(d, ts=opts["Input"]["indexCol"])

        #Fill in missing rows
        #(Requires 'd =' because reindexing cannot be done in place if rows are added)
        d = fillMissing(d, freq=opts["Input"]["baseFreq"], indexCol=opts["Input"]["indexCol"])

        #Save Level 1 data
        saveOutput(d, opts["Output"]["outputPath"], fname, "_Level1", outputNans=opts["Output"]["outputNans"], index=False)

        if opts["Level 1"]["altCheckEnable"]:
            d_dev = calcAltDevs(d, fname, **opts["Level 1"], **opts["Input"])
            saveOutput(d_dev, opts["Output"]["outputPath"], fname, "_Level1_dev", outputNans=opts["Output"]["outputNans"], index=False)
            #devMask = createDevMask(d_dev, opts["Level 1"]["devThreshold"], sourceChars)
        else:
            d_dev = None

        logCurrentFile = "[" + os.path.split(fname)[1] + "]\t"

        #Create and save QA dataframe
        d_qa = createQAdf(d, sourceChars, opts["Input"]["indexCol"], devThreshold=opts["Level 1"]["devThreshold"], d_dev=d_dev)

        #Save QA data
        saveOutput(d_qa, opts["Output"]["outputPath"], fname, "_Level1_qa", outputNans=opts["Output"]["outputNans"], indexCol="idx")

        #Calc QA stats
        d_stats = calcQAStats(d, d_qa, d_dev)

        #Save QA stat data
        saveOutput(d_stats, opts["Output"]["outputPath"], fname, "_Level1_qa_stats", outputNans=opts["Output"]["outputNans"])

    logCurrentFile = ""
    logging.info("Done.")