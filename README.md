# CSVQA
 Tools for QA/QC checks and fixes for CSV files

## Workflow

|Data State|Input|Output|Procedure|
|-|-|-|-|
|Level 0   |  Data acquisition   |   Raw data   | N/A
|Level 1   | - Primary data as CSV(s): *filename.csv*<BR>- Configuration file: *default.conf*<BR>Optionally:<BR>- Source characteristics file<BR>- Alternate data source files(s): *filename_alt.csv* | -Level 1 data: *filename_Level1.csv* <BR>- QA mask file: *filename_Level1_qa.csv*<BR>- Deviation file:*filename_Level1_dev.csv*<BR>- Log file| Level0to1.py|
|Level 2   | - Level 1 data: *filename_Level1.csv*<BR>- Configuration file: *default.conf*<BR>- Reviewed QA mask file: *filename_Level2_mask.csv* <BR>- Deviation file: *filename_Level1_dev.csv* | - Processed, combined\* data: *filename_Level2.csv*<BR>- Data aggregated to additional intervals: *filename_Level2_avg_interval.csv* | Level1to2.py |
|Level 3   | | | | 

<I>\*(If alternate source included)</I>

### Recommended Procedure:
1. Prepare Sources
    * Prepare source_characteristics.csv
    * Format any alternate data to have the same column names and filename appended with "_alt" as the primary source
    * Create a copy of default.conf for the specific data run and make any necessary changes
    
2. Run Level0to1.py

3. Review Output
    * *filename_Level1_qa_stats.csv* is useful to determine where to look for issues, by looking at the QA mask counts and stats and for each field.
    * *filename_Level1_dev.csv* is useful to compare primary and alternate sources.
    * *filename_Level1_qa.csv* should be modified to match the intended QA actions.  This should include replacing any combination flags.
    * *filename_Level1.csv* is always good reference for checking what data raised flags.
    
4. Run Level1to2.py.

If an alternate data source uses a larger sample interval than the primary source, values will be back-filled.  An alternate source with a shorter sample interval than the primary source may have issues (has not been tested).

**Warning:** These scripts WILL overwrite files of the same name in the output directory.  If an existing file is locked, it will append an integer to the end of the file name.

**Other Notes:**

In the future, the functions used in these scripts should be made into a proper object-oriented class.  This would also simplify the number of variables that need to be passed around globally by letting them exist privately in the class.

These scripts are not optimized for memory usage and should be expected to potentially use more than twice the memory as the size of the original data, possibly much more depending on resampling intervals.  They are also not optimized for multi-threaded processing; an obvious improvement could be made by run resampling of different intervals in parallel.

**Known Issues:**
- If timestamp column isn't first, resampling may fail.
- Index column is not excluded from correlation calcs, throws a warning but doesn't affect output.

### QA Mask

After running Level0to1.py, the output should be manually reviewed in preparation for running Level1to2.py.  Specifically, the QA mask file should be updated to reflect the desired changes for Level 2.

The QA mask file contains the following flags for each data point, which will indicate the corresponding actions to take in Level1to2.py.

|Value|Data Status|Action|
|-|-|-|
|0|Good|Keep as-is|
|1|Out of dat/time range|Remove|
|2|Out of source characteristic range|Replace with Nan value|
|3|1 + 2|None|
|4|Significant deviation from alternate source|Replace with alternate source value|
|5|1 + 4|None|
|6|2 + 4|None|
|7|1 + 2 + 4|None|

An easy way to check and modify the flagged areas is to open the mask file in Excel and filter by the column and flag of interest.  Affected values can then easily be selected and changed in bulk.  The modified file should be saved as ***filename_Level2_mask.csv***.  Be sure that the Timestamp format in Excel is set to "yyyy-mm-dd hh:mm:ss" before saving.  Combination flags have no effect on the output and should be manually replaced to reflect the intended action.

### Source Characteristics

This should be a CSV with information about the data sources.  Ideally, one row should be used for each type of measurement.  The fields currently implemented are:
|Field|Function|Format|
|-|-|-|
|"Min"|Minimum expected value, used to classify QA flag "2"|Float|
|"Max"|Maximum expected value, used to classify QA flag "2"|Float|
|"Useful Decimal Places"|Decimal places to use in the final output|Integer|
|"Columns"|Determines which source column(s) the parameters in the row should apply to.|String(s), separated by ';' for multiple|
|"Start", Timestamp<BR>(In first, second columns of any row)|Starting date/time of dataset, used to classify QA flag "1"|String,Timestamp| 
|"End", Timestamp<BR>(In first, second columns of any row)|Ending date/time of dataset, used to classify QA flag "1"|String,Timestamp| 

Additional columns can be added as reference without any effect.


******************************

### Level0to1.py
#### Tool for processing data from Level 0 to Level 1
See parameters defined in configuration file (default.conf) for details.

The output of this should be manually reviewed before being used for Level 2.
******************************
### Level1to2.py
#### Tool for processing data from Level 1 to Level 2
See parameters defined in configuration file (default.conf) for details.


## Other Utilities:

### csv_avg.py
#### Tool for averaging data to one or more timescales
See parameters defined in script for details.


## Example Data
### /data
"Chamber1_2020_sample_1Min.dat" is a modified snippet of a data file, intentionally modified to have issues.  The other files are the output generated by csv_avg.py using the default configuration.

### /data/2018
|File|Description|
|-|-|
|Met Station_Min1.dat\*|Raw data as CSV|
|Met Station_Min1_alt.dat\*|Alternate data source for "AirT" column|
|source_characteristics.csv|Source characteristics file|

\*Not yet uploaded


Zenodo archive: <br>
[![DOI](https://zenodo.org/badge/300742022.svg)](https://zenodo.org/badge/latestdoi/300742022)
