import pandas as pd

class Query:
    """
    One line of query request. Contains:
    tables: a dictionary, alias -> tableName
    connect: an array of couples, contains couple metas like tableName.keyName (couple[0] < couple[1])
    condition: an dictionary, tableName.keyName_[lb/ub] -> Value
    result: an integer, -1 if none
    """
    def __init__(self, str):
        temp = str.split('#')
        if len(temp) < 3:
            raise Exception("Error when init the class Query")
        self.tablesStr = temp[0]
        self.connectStr = temp[1]
        self.conditionStr = temp[2]
        self.resultStr = None if len(temp) == 3 else temp[3]
        self.tables = self.handleTables(self.tablesStr)
        self.connect = self.handleConnect(self.connectStr)
        self.condition = self.handleCondition(self.conditionStr)
        self.result = self.handleResult(self.resultStr)

    def handleTables(self, str):
        temp = str.split(',')
        dic = {}
        for each in temp:
            a, b = each.split(' ')
            dic[b] = a
        return dic

    def handleConnect(self, str):
        if str == '':
            return []
        temp = str.split(',')
        arr = []
        for each in temp:
            a, b = each.split('=')
            aTable, aKey = a.split('.')
            bTable, bKey = b.split('.')
            if aTable in self.tables:
                aTable = self.tables[aTable]
            if bTable in self.tables:
                bTable = self.tables[bTable]
            a = aTable + '.' + aKey
            b = bTable + '.' + bKey
            if a < b:
                arr.append([a, b])
            else:
                arr.append([b, a])
        return arr

    def handleCondition(self, str):
        temp = str.split(',')
        # Change all alias to original names
        for i in range(len(temp)):
            ttemp = temp[i].split('.')
            if len(ttemp) == 2:
                if ttemp[0] in self.tables:
                    temp[i] = self.tables[ttemp[0]] + '.' + ttemp[1]

        i = 0
        dic = {}
        while i < len(temp):
            a = temp[i]
            b = temp[i+1]
            c = temp[i+2]
            # switch(operator)
            if b == '=':
                dic[a + '_lb'] = int(c)
                dic[a + '_ub'] = int(c)
            elif b == '<':
                dic[a + '_ub'] = int(c)
            elif b == '>':
                dic[a + '_lb'] = int(c)
            i += 3
        return dic

    def handleResult(self, str):
        if str is None:
            return -1
        else:
            return int(str)

# Test Point A -- OK
# with open('handout/train.csv') as file:
#     line = file.readline()
#     test = Query(line)
#     print('done')

def coupleToStr(couple):
    """
    Transform a 2-dim array to string
    e.g.
    ["t.id", "b.id"] => t.id;b.id
    """
    return couple[0] + ';' + couple[1]

def generateOneX(sigmaConn, sigmaCond, query):
    """
    Transform a query to a Vector
    query => [connection:condition]
    """
    connections = [0 for i in range(len(sigmaConn.keys()))]
    conditions = [0 for i in range(len(sigmaCond.keys()))]

    for conn in query.connect:
        connections[sigmaConn[coupleToStr(conn)]] = 1

    for cond in query.condition.keys():
        conditions[sigmaCond[cond]] = query.condition[cond]

    return connections + conditions

def generateX(queries):
    """
    Input: a list of Query objects
    Output: sigmaConn, sigmaCond, X
    """
    
    # First, get the One-Heat encoding dictionary for connections
    sigmaConnArr = []
    for each in queries:
        for i in each.connect:
            if i not in sigmaConnArr:
                sigmaConnArr.append(i)
    sorted(sigmaConnArr, key=lambda x: coupleToStr(x))
    sigmaConn = {}
    for i in range(len(sigmaConnArr)):
        sigmaConn[coupleToStr(sigmaConnArr[i])] = i

    # Second, get the One-Heat encoding dictionary for conditions
    sigmaCondArr = []
    for each in queries:
        for key in each.condition.keys():
            if key not in sigmaCondArr:
                sigmaCondArr.append(key)
    sorted(sigmaCondArr)
    sigmaCond = {}
    for i in range(len(sigmaCondArr)):
        sigmaCond[sigmaCondArr[i]] = i

    # For Debug
    # print("sigmaConn:")
    # print(sigmaConn)
    # print("sigmaCond:")
    # print(sigmaCond)

    # generate code for each element
    X = []
    for each in queries:
        X.append(generateOneX(sigmaConn, sigmaCond, each))

    return [sigmaConn, sigmaCond, X]

def generatePredX(sigmaConn, sigmaCond, queries):
    X = []
    for each in queries:
        X.append(generateOneX(sigmaConn, sigmaCond, each))
    return X

def generateY(queries):
    Y = []
    for each in queries:
        Y.append(each.result)
    return Y

# Test Point B -- OK
queries = []
predQueries = []
with open('handout/train.csv') as file:
    line = file.readline()
    while line != '':
        queries.append(Query(line))
        line = file.readline()
with open('handout/test_without_label.csv') as file:
    line = file.readline()
    while line != '':
        predQueries.append(Query(line))
        line = file.readline()

# Notice: All limits in testing set are in training set

# Test Point C -- Ok
Xtemp = generateX(queries)
sigmaConn = Xtemp[0]
sigmaCond = Xtemp[1]
X = Xtemp[2]
Y = generateY(queries)
print('Done')

# Test Point D -- Ok
Xpred = generatePredX(sigmaConn, sigmaCond, predQueries)
print('Done')
