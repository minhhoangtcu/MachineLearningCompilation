import numpy as np

def pandas2arff(df,filename,wekaname = "pandasdata",cleanstringdata=True,cleannan=True):
    """
    converts the pandas dataframe to a weka compatible file
    df: dataframe in pandas format
    filename: the filename you want the weka compatible file to be in
    wekaname: the name you want to give to the weka dataset (this will be visible to you when you open it in Weka)
    cleanstringdata: clean up data which may have spaces and replace with "_", special characters etc which seem to annoy Weka.
                     To suppress this, set this to False
    cleannan: replaces all nan values with "?" which is Weka's standard for missing values.
              To suppress this, set this to False
    """
    import re

    def cleanstring(s):
        if s!="?":
            return re.sub('[^A-Za-z0-9]+', "_", str(s))
        else:
            return "?"
        
    def clean_movie(s):
        return re.sub("'+", "", str(s))

    dfcopy = df #all cleaning operations get done on this copy


    if cleannan!=False:
        dfcopy = dfcopy.fillna(-999999999) #this is so that we can swap this out for "?"
        #this makes sure that certain numerical columns with missing values don't get stuck with "object" type

    f = open(filename,"w")
    arffList = []
    arffList.append("@relation " + wekaname + "\n")
    #look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
    for i in range(df.shape[1]):
        movie = clean_movie(str(df.iloc[0, i]))
        arffList.append("@attribute '" + movie + "' NUMERIC\n")
        
    arffList.append("@data\n")
    
    _instanceString = ""
    for j in range(dfcopy.shape[1]):
        _instanceString+=str(dfcopy.iloc[1, j])
        
        if j != dfcopy.shape[1] - 1:#if it's not the last feature, add a comma
            _instanceString += ","
        
    _instanceString += "\n"
            
    arffList.append(_instanceString)
    f.writelines(arffList)
    f.close()
    del dfcopy
    return True
