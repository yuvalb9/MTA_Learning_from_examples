
import random
import string
import os
import pandas as pd

class DataFactory:
    def  __init__(self, goodvaluespath, alphabet=string.ascii_letters+string.digits+" "):
        self.valid = []
        self.invalid = []
        self.alphabet = alphabet

        if not os.path.exists(goodvaluespath):
            raise ValueError("goodvaluespathis not a valid path")

        filenames = os.listdir(goodvaluespath)
        for filename in filenames:
            currfullpath = os.path.join(goodvaluespath, filename)
            with open(currfullpath, "r") as fp:
                for line in fp:
                    cleanedline = self.cleanline(line)
                    if cleanedline not in self.valid:
                        self.valid.append( cleanedline )

        self.generateinvalids()


    def cleanline(self, rawinput):
        temp = rawinput.strip()
        return temp

    def getvalid(self):
        return self.valid.copy()


    def getinvalid(self):
        return self.invalid.copy()

    def generateinvalids(self):
        for goodword in self.valid:
            badword = "".join([random.choice(self.alphabet) for ch in goodword])
            while badword in self.invalid:
                badword = "".join([random.choice(self.alphabet) for ch in goodword])

            self.invalid.append(badword)


def statistics(predicitions, targets):
    stats = {'TP' :0 ,'FP' :0 ,'TN' :0 ,'FN' :0}
    for i in range(len(predicitions)):
        isBad    = targets[i, 1] > 0.5
        isTagged = predicitions[i ,1] > 0.5
        if isBad and isTagged:
            stats['TP'] += 1
        elif (isBad) and (not isTagged):
            stats['FN' ]+=1
        elif (not isBad) and (isTagged):
            stats['FP' ]+=1
        elif (not isBad) and (not isTagged):
            stats['TN' ]+=1

    stats['PR'] = (1.00000000 *stats['TP']) / (stats['FP'] +stats['TP'] +1)
    stats['RE'] = (1.00000000 *stats['TP']) / (stats['FN'] +stats['TP'] +1)
    return stats


def get_dataframe():
    factory = DataFactory("StreetNames/", alphabet=string.ascii_letters + string.digits + " ")
    validDF = pd.DataFrame({'word': factory.getvalid()})
    validDF['tag'] = 0
    invalidDF = pd.DataFrame({'word': factory.getinvalid()})
    invalidDF['tag'] = 1
    ret = pd.DataFrame( validDF )
    ret = ret.append( invalidDF, ignore_index=True )
    ret = ret.sample(frac=1)
    ret = ret.reset_index(drop=True)
    return ret


def get_histogram(df):
    histogram = {}
    #df = pd.DataFrame()
    i = 0
    for row in df[ df['tag']==0 ]['word']:
        for ch in row:
            if not ch in histogram:
                histogram[ch] = 0
            histogram[ch] += 1

    return histogram



def get_percentage(histogram):
    sum_appearances = sum([v for k, v in histogram.items()])
    percentage = {}
    for k, v in histogram.items():
        percentage[k] = ((1.000000 * v) / sum_appearances)
    return percentage


def transform_df(df, percentage):
    for i in range(21):
        df["ch{0}".format(i + 1)] = df['word'].str[i]
        df["pr{0}".format(i + 1)] = df['word'].str[i].map(percentage)


def calculate_precision_cutoff(pred_float, tag):
    n_samples = pred_float.shape[0]
    cutoffs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    cutoff_data = {}
    for cutoff in cutoffs:
        cutoff_data[cutoff] = {"TP":0.0, "FP":0.0, "TN":0.0, "FN":0.0 , "TOTAL":n_samples}
        for i in range(n_samples):
            if (pred_float[i, 0] > cutoff) and (tag[i] ==1):
                cutoff_data[cutoff]["TP"] += 1.0
            elif (pred_float[i, 0] > cutoff) and (tag[i] == 0):
                cutoff_data[cutoff]["FP"] += 1.0
            elif (pred_float[i, 0] <= cutoff) and (tag[i] == 1):
                cutoff_data[cutoff]["FN"] += 1.0
            elif (pred_float[i, 0] <= cutoff) and (tag[i] == 0):
                cutoff_data[cutoff]["TN"] += 1.0

        try:
            cutoff_data[cutoff]["PR"] = cutoff_data[cutoff]["TP"] / (cutoff_data[cutoff]["TP"] + cutoff_data[cutoff]["FP"])
        except ZeroDivisionError:
            cutoff_data[cutoff]["PR"] = 0
        try:
            cutoff_data[cutoff]["RE"] = cutoff_data[cutoff]["TP"] / (cutoff_data[cutoff]["TP"] + cutoff_data[cutoff]["FN"])
        except ZeroDivisionError:
            cutoff_data[cutoff]["RE"] = 0
        try:
            cutoff_data[cutoff]["AC"] = (cutoff_data[cutoff]["TP"]+cutoff_data[cutoff]["TN"]) / (n_samples)
        except ZeroDivisionError:
            cutoff_data[cutoff]["AC"] = 0

    return cutoff_data

def learn_naive_dnn(new_df):
    """:type new_df: pd.DataFrame """
    from keras.models import Sequential
    from keras.optimizers import SGD, Adam
    from keras.layers.core import Dense, Dropout
    X = new_df[['pr1', 'pr2', 'pr3', 'pr4', 'pr5', 'pr6', 'pr7', 'pr8', 'pr9', 'pr10', 'pr11', 'pr12', 'pr13', 'pr14', 'pr15', 'pr16', 'pr17', 'pr18', 'pr19', 'pr20', 'pr21']].values
    Y = new_df['tag'].values

    # Set constants
    batch_size = 128
    dimof_input = 21
    dimof_middle = 20
    dimof_output = 1
    dropout = 0.2
    countof_epoch = 40
    verbose = False
    print('batch_size: ', batch_size)
    print('dimof_middle: ', dimof_middle)
    print('dropout: ', dropout)
    print('countof_epoch: ', countof_epoch)

    print('verbose: ', verbose)
    print()

    # Set model
    model = Sequential()
    model.add(Dense(dimof_middle, input_dim=dimof_input, init='uniform', activation='sigmoid'))
    model.add(Dense(dimof_middle, init='uniform', activation='sigmoid'))
    model.add(Dense(dimof_output, init='uniform', activation='sigmoid'))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    print (model.summary())
    # Train

    model.fit(
        X, Y,
        shuffle=True,
        validation_split=0.2,
        batch_size=batch_size, nb_epoch=countof_epoch, verbose=verbose)

    x = model.predict(X)
    d = calculate_precision_cutoff(x, Y)
    for k in d.keys():
        print ("cutoff: ", k, " PR: ", d[k]["PR"], " RE: ", d[k]["RE"], " AC: ", d[k]["AC"])


def main():
    df = get_dataframe()
    histogram = get_histogram(df)
    percentage = get_percentage(histogram)
    transform_df(df, percentage)
    new_df = df[['pr1', 'pr2', 'pr3', 'pr4', 'pr5', 'pr6', 'pr7', 'pr8', 'pr9', 'pr10', 'pr11', 'pr12', 'pr13', 'pr14', 'pr15', 'pr16', 'pr17', 'pr18', 'pr19', 'pr20', 'pr21', 'tag']]
    new_df.fillna(value=0, inplace=True)

    learn_naive_dnn(new_df)


if __name__ == '__main__':
    main()














