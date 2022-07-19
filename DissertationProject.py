import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import streamlit as st 
import requests
from pandas_datareader import data
from pandas_datareader.data import DataReader
from datetime import datetime
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

image = Image.open('download.jfif')




dashboard=st.sidebar.selectbox("Select DashBoard",('Index Prediction','Stock Price Prediction'))
st.header(dashboard)
if dashboard == 'Index Prediction':
    st.image(image, caption='DOJIA Stock-Sentiment-Analysis',use_column_width=True)
    data1 = pd.read_csv(r"C:\Users\manis\OneDrive\Documents\Python\Data.csv", encoding = "ISO-8859-1")
    st.write('In this DashBoard we will try to predict the stock market index DOW JONES INDUSTRIAL AVERAGE (commonly known as DOJIA) movement based on Daily news.')
    st.write(data1.head())
    st.subheader('Checking if DataSet has any null Values')
    st.write(data1.isnull().sum().tail())
    st.subheader('Filling the null values with median Values') 

    st.write('data1[''Top23''].fillna(data1[''Top23''].median,inplace=True)')
    st.write('data1[''Top24''].fillna(data1[''Top24''].median,inplace=True)')
    st.write('data1[''Top25''].fillna(data1[''Top25''].median,inplace=True)')
    
    symbol=st.sidebar.text_input('Index Symbol',value='^DJI',max_chars=11)
    
    end=datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    st.write(start,end)
    dataset = DataReader(symbol, 'yahoo', start, end)
    st.subheader('Live '+ symbol+ ' Data read from Yahoo Finance')
    st.write(dataset)
    pd=dataset.drop(["Volume","Adj Close"],axis=1)
    #pd=dataset.drop(["Adj Close"],axis=1)
    chart_data = pd
    st.subheader('Live '+ symbol+ ' Chart from the Yahoo Finance Data')
    st.line_chart(chart_data)
    
    
    
    stock=requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")
    StockData=stock.json()
    
    #for message in StockData['messages']:
    #   st.subheader(message['user']['username'])
    #   st.image(message['user']['avatar_url'])
    #   st.write(message['body'])
    #   st.write(message['created_at'])
        
        
    #enddate = dt.datetime.today()
    #begdate = enddate + relativedelta(years=-1)
    #x = pdr.get_data_google(".DJI",begdate,enddate)
    #st.write(x)

    train = data1[data1['Date'] < '20150101']
    test = data1[data1['Date'] > '20141231']

    st.subheader('Removing punctuations and changing all the letters to lowercase for both train and test')
    
    all_data = [train,test]

    for df in all_data:
        df.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
        for i in df.columns:
            if i=='Date':
                continue
            if i=='Label':
                continue
            df[i] = df[i].str.lower()

    st.write(train.head())
    st.subheader('Combining all the news')
    headlines = []
    for row in range(0,len(train.index)):
        headlines.append(' '.join(str(x) for x in train.iloc[row,2:]))
    headlines[0]

    # combining all the headlines in test data into one and appending them into a list 

    test_transform= []
    for row in range(0,len(test.index)):
        test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
    #     print(test_transform[x])

    # Applying countvectorizer on headlines list that we created before and max features is set to 100009
    st.subheader('Vectorization')
    st.write('As is well know Machine learning algorithms only work with numbers, therefore these news articles need to be converted into numbers so that the algorithms can understand the news polarity and sentiments. This exactly can be achieved by CountVectorizer in machine learning.    The method of converting a series of text documents into numerical function vectors is known as Vectorization.')
    st.write('Tokenization: By using white-spaces and punctuation as token separators, you can tokenize strings and assign an integer id to each possible token. Counting: Token occurrences in each text are counted.')
    st.write('Normalization: Tokens that appear in the majority of samples / documents are normalised and weighted with decreasing value tokens.')

    
    if st.button('Run Algorithm Accuracy Comparison'):
            # move begin
            countvector=CountVectorizer(ngram_range=(2,2),max_features=800)
            traindataset=countvector.fit_transform(headlines)

            randomclassifier=RandomForestClassifier(n_estimators=20,criterion='entropy')
            randomclassifier.fit(traindataset,train['Label'])
            # Applying countvectorizer on test_transform list that we created before 

            test_dataset = countvector.transform(test_transform)
            st.subheader('Predicting by RandomForestClassifier')
            predictions = randomclassifier.predict(test_dataset)
            # print(predictions)
            # accuracy score (compared test daset original output values with predictions)
            score=accuracy_score(test['Label'],predictions)
            st.write('RandomForestClassifier: ',round(score*100,2),'%')
            st.write('===============================')
            matrix=confusion_matrix(test['Label'],predictions)
            st.subheader('confusion matrix : ')
            import seaborn as sns
            fig5 = plt.figure(figsize=(6,3))
            sns.heatmap(matrix , annot=True , fmt=".1f",xticklabels=['Market Moves Down' , 'Market Moves Up'] , yticklabels=['Market Moves Down' , 'Market Moves Up'])
            plt.ylabel("True")
            plt.xlabel("Predicted")
            st.pyplot(fig5)
            
            #import seaborn as sns
            #sns.heatmap(matrix, annot=True)
            st.write(matrix)
            
            CReport = classification_report(test['Label'],predictions)
            st.write(CReport)
            #moved
            def my_dict(name,score):
                my_matrix = {}
                my_matrix['Algorithm Name'] = name
                my_matrix['Score'] = str(round((score*100),2 ))+'%'
                return my_matrix
            import pandas as pd    
            resultsdf = pd.DataFrame({'A' : []})
            resultsdf = resultsdf.from_dict(my_dict('RandomForestClassifier',score),orient = 'index')
            resultsdf=resultsdf.transpose()
            #st.write(resultsdf)
            max_features_num = [800]
            ngram = [2,2]

            for i in max_features_num:
                for j in ngram:
                    countvector=CountVectorizer(ngram_range=(j,j),max_features=i)
                    traindataset=countvector.fit_transform(headlines)
                    test_dataset = countvector.transform(test_transform)
                    xgb = XGBClassifier(random_state =1,eval_metric = "error")
                    xgb.fit(pd.DataFrame(traindataset.todense(), columns=countvector.get_feature_names()),train['Label'])
                    predictions = xgb.predict(pd.DataFrame(test_dataset.todense(), columns=countvector.get_feature_names()))
                    score=accuracy_score(test['Label'],predictions)
                    print('XGBClassifier')
                    print('max number of features used : {}'.format(i))
                    print('ngram_range ({},{})'.format(j,j))
                    print(score)
                    matrix=confusion_matrix(test['Label'],predictions)
                    print('confusion matrix : {}'.format(matrix))
                    print('===============================')
            resultsdf = resultsdf.append(my_dict('XGBClassifier',score),ignore_index = True)
            #st.write(resultsdf)
            cb=CatBoostClassifier(random_state=1)
            cb.fit(pd.DataFrame(traindataset.todense(), columns=countvector.get_feature_names()),train['Label'])
            predictions = xgb.predict(pd.DataFrame(test_dataset.todense(), columns=countvector.get_feature_names()))
            matrix=confusion_matrix(test['Label'],predictions)
            score=accuracy_score(test['Label'],predictions)
            resultsdf = resultsdf.append(my_dict('CatBoostClassifier',score),ignore_index = True)
            #st.write(resultsdf)
            from sklearn.tree import DecisionTreeRegressor
            import seaborn as sns

            #Fit x_train and y-train into the regression model
            #fitting the decision tree regression model to the dataset without splitting the dataset
            regressor=DecisionTreeRegressor(random_state=0)
            regressor.fit(traindataset,train['Label'])

            y_pred = regressor.predict(test_dataset)
            matrix=confusion_matrix(test['Label'],y_pred)
            score=accuracy_score(test['Label'],y_pred)
            resultsdf = resultsdf.append(my_dict('DecisionTreeRegressor',score),ignore_index = True)
            #st.write(resultsdf)
            from sklearn.linear_model import LogisticRegression
            model_L=LogisticRegression()
            model_L.fit(traindataset,train['Label'])
            y_pred = model_L.predict(test_dataset)
            score=accuracy_score(test['Label'],y_pred)
            resultsdf = resultsdf.append(my_dict('LogisticRegression',score),ignore_index = True)
            #st.write(resultsdf)
            from sklearn.svm import SVC
            model_SVM=SVC()
            model_SVM.fit(traindataset,train['Label'])
            y_pred = model_SVM.predict(test_dataset)
            matrix=confusion_matrix(test['Label'],y_pred)
            score=accuracy_score(test['Label'],y_pred)
            resultsdf = resultsdf.append(my_dict('SupportVectorMachine',score),ignore_index = True)
            #st.write(resultsdf)
            from sklearn.neighbors import KNeighborsClassifier
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(traindataset,train['Label'])
            y_pred = neigh.predict(test_dataset)
            matrix=confusion_matrix(test['Label'],y_pred)
            score=accuracy_score(test['Label'],y_pred)
            resultsdf = resultsdf.append(my_dict('KNeighborsClassifier',score),ignore_index = True)
            #st.write(resultsdf)
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(traindataset.todense(),train['Label'])

            y_pred = model.predict(test_dataset.todense())
            matrix=confusion_matrix(test['Label'],y_pred)
            score=accuracy_score(test['Label'],y_pred)
            #print('GaussianNB')
            #print(score*100,"%")
            #print('===============')
            #print(matrix)
            resultsdf = resultsdf.append(my_dict('GaussianNB',score),ignore_index = True)
            st.write(resultsdf)
            
            for i in resultsdf:
                resultsdf.replace("[%]"," ",regex=True, inplace=True)
            resultsdf["Score"] = pd.to_numeric(resultsdf["Score"])
            

            
            import matplotlib.pyplot as plt

            # Pie chart, where the slices will be ordered and plotted counter-clockwise:
            labels = resultsdf['Algorithm Name']
            sizes = resultsdf['Score']
            explode = (.1, .2, 0,.2,0,.2,0,.2)  # only "explode" the 2nd slice (i.e. 'Hogs')

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(fig1)
            
            import matplotlib.pyplot as plt
            

            # Fixing random state for reproducibility
            #np.random.seed(19680801)
            np.random.seed(0)

            plt.rcdefaults()
            fig, ax = plt.subplots()

            # data
            people = resultsdf['Algorithm Name']
            y_pos = np.arange(len(people))
            performance = resultsdf['Score']
            error = np.random.rand(len(people))

            ax.barh(y_pos, performance, xerr=error, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(people)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Performance')
            ax.set_title('Prediction Accuracy Scores for Machine Learning Algorithms')
            for i, v in enumerate(performance):
                ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
            plt.show()
            st.pyplot(fig)
            st.subheader('XGBClassifier and CatBoostClassifier both performed very well a highest accuracy score of 84.66%. While SupportVectorMachine came a close second with score of 84.39')
    
if dashboard == 'Stock Price Prediction':
    
    # stock can be fed via the imput control on the page
    symbol=st.sidebar.selectbox("Stock Symbol",('AAPL','JPM','NKE','MSFT'))
    # Setting up image as page header
    st.image(image, caption='USA S&P500 Stock-Price-Prediction',use_column_width=True)
    #setting up the timeframe for the stock data 
    end=datetime.now()
    start='2012-01-01'
    # calling the datareader to fetch the data from yahoo finance 
    dataset = DataReader(symbol, 'yahoo', start, end)
    st.subheader('Live '+ symbol+ ' Data read from Yahoo Finance')
    st.write(dataset)
    # Droping the ‘volume’ collumn for the price chart plot
    pd=dataset.drop(["Volume"],axis=1)
    # Visualizing the Close price
    plt.figure(figsize=(16,8))
    plt.title(symbol +' Close Price History')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(pd['Close'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.image(f"https://charts2.finviz.com/chart.ashx?t={symbol}")
    
    # Visualizing all columns in the dataset
    for i in dataset.columns: 
        names = i
        values = dataset[i]
        plt.figure(figsize=(16, 8))
        plt.subplot()
        plt.plot(dataset.index, values,c='r')
        plt.suptitle(i, fontsize=18)
        plt.xlabel('Date', fontsize=18)
        plt.show()
        
    axes = dataset.plot(figsize=(10, 10),rot=0, color = ('r','b', 'g','y','c','m' ), subplots=True)
    axes[1].legend(loc=2) 
    st.pyplot()
    # Moving code
    def my_dict(Stock,name,Error,score):
        my_matrix = {}
        my_matrix['Algorithm Name'] = name
        my_matrix['Stock Name'] = Stock
        
        my_matrix['Error Type'] = Error
        my_matrix['Score'] = score
        return my_matrix
    import pandas as pd
    
    resultsdf = pd.DataFrame()
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    df=dataset[['Close']]
    future_days = 100
    df['Predicted']=df['Close'].shift(-future_days)
    X=np.array(df.drop(['Predicted'],1))[:-future_days]
    y=np.array(df['Predicted'])[:-future_days]
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.30)
    
    lr=LinearRegression().fit(x_train, y_train)
    x_future = df.drop(['Predicted'],1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    lr_prediction=lr.predict(x_future)
    pred = lr_prediction
    valid=df[X.shape[0]:]
    valid['prediction']=pred
    plt.figure(figsize=(15,8))
    plt.title(symbol+' Stock Prices 2012-Today by Linear Regression')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.plot(df['Close'])
    plt.plot(valid[['Close','prediction']])
    plt.legend(['Training ML-Model','Actual Close Price','Predicted Close Price'])

    resultsdf=  resultsdf.append(my_dict(symbol+':','Linear Regression:','R-Square Error:',metrics.r2_score(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Linear Regression:','Mean Absolute Error:', metrics.mean_absolute_error(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Linear Regression:','Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(valid['Close'],valid['prediction']))),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Linear Regression:','Mean Squared Error:', metrics.mean_squared_error(valid['Close'],valid['prediction'])),ignore_index=True)
    st.pyplot()
    
    from sklearn.ensemble import RandomForestRegressor
    RR=RandomForestRegressor()
    RR.fit(x_train, y_train)
    x_future = df.drop(['Predicted'],1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    lr_prediction=RR.predict(x_future)
    pred = lr_prediction
    valid=df[X.shape[0]:]
    valid['prediction']=pred
    plt.figure(figsize=(15,8))
    plt.title(symbol+' Stock Prices 2012-Today by Random Forest Regressor')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.plot(df['Close'])
    plt.plot(valid[['Close','prediction']])
    plt.legend(['Training ML-Model','Actual Close Price','Predicted Close Price'])

    resultsdf=  resultsdf.append(my_dict(symbol+':','Random Forest Regressor:','R-Square Error:',metrics.r2_score(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Random Forest Regressor:','Mean Absolute Error:', metrics.mean_absolute_error(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Random Forest Regressor:','Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(valid['Close'],valid['prediction']))),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Random Forest Regressor:','Mean Squared Error:', metrics.mean_squared_error(valid['Close'],valid['prediction'])),ignore_index=True)
    st.pyplot()
    

    from sklearn.tree import DecisionTreeRegressor

    #Fit x_train and y-train into the regression model
    #fitting the decision tree regression model to the dataset without splitting the dataset
    DT=DecisionTreeRegressor(criterion='mse',
    splitter='best',
    max_depth=10,
    min_samples_split=8,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    presort='deprecated',
    ccp_alpha=0.0,)
    
    
    
    DT.fit(x_train, y_train)
    x_future = df.drop(['Predicted'],1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    lr_prediction=RR.predict(x_future)
    pred = lr_prediction
    valid=df[X.shape[0]:]
    valid['prediction']=pred
    plt.figure(figsize=(15,8))
    plt.title(symbol+' Stock Prices 2012-Today by Decision Tree')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.plot(df['Close'])
    plt.plot(valid[['Close','prediction']])
    plt.legend(['Training ML-Model','Actual Close Price','Predicted Close Price'])

    resultsdf=  resultsdf.append(my_dict(symbol+':','Decision Tree:','R-Square Error:',metrics.r2_score(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Decision Tree:','Mean Absolute Error:', metrics.mean_absolute_error(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Decision Tree:','Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(valid['Close'],valid['prediction']))),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Decision Tree:','Mean Squared Error:', metrics.mean_squared_error(valid['Close'],valid['prediction'])),ignore_index=True)
    st.pyplot()
    
    
    from sklearn.svm import SVR
    SVregressor = SVR(kernel='rbf')
    SVregressor.fit(x_train, y_train)
    x_future = df.drop(['Predicted'],1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    lr_prediction=SVregressor.predict(x_future)
    pred = lr_prediction
    valid=df[X.shape[0]:]
    valid['prediction']=pred
    plt.figure(figsize=(15,8))
    plt.title(symbol+' Stock Prices 2012-Today by Support Vector Regressor')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.plot(df['Close'])
    plt.plot(valid[['Close','prediction']])
    plt.legend(['Training ML-Model','Actual Close Price','Predicted Close Price'])

    resultsdf=  resultsdf.append(my_dict(symbol+':','Support Vector Regressor:','R-Square Error:',metrics.r2_score(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Support Vector Regressor:','Mean Absolute Error:', metrics.mean_absolute_error(valid['Close'],valid['prediction'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Support Vector Regressor:','Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(valid['Close'],valid['prediction']))),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','Support Vector Regressor:','Mean Squared Error:', metrics.mean_squared_error(valid['Close'],valid['prediction'])),ignore_index=True)
    st.pyplot()
    
    
    
    
    
    
    
    

    
    
          
    # Creating dataframe with 'Close' column 
    data = dataset.filter(['Close'])
    # Converting the dataframe to a numpy array
    NPdataset = data.values
    
    # Get the data for training the model
    train_len = int(np.ceil( len(NPdataset) * .70 ))
    
    
    # using minmax sclaer to Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    rescaled_dSet = scaler.fit_transform(NPdataset)
    
    # Creating training data  
    # Creating rescaled training data set
    train_data = rescaled_dSet[0:int(train_len), :]
    # Spliting dataset into x_train and y_train 
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshaping data for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    # Building LSTM sequential prediction model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compiling the LSTM model
    model.compile(optimizer='adam', loss='mean_squared_error')
    import os
    from keras.models import Sequential, load_model
    # Training the model and provisioning for training save
    if(not os.path.exists('S&P500_prediction.h5')):
        model.fit(x_train, y_train, batch_size=12 , epochs=15)
        model.save('S&P500_prediction.h5')
    model = load_model('S&P500_prediction.h5')
    # model.fit(x_train, y_train, batch_size=1, epochs=1)
    # testing set creation 
    # new array for rescaled values from index 1543 to 2002 
    test_data = rescaled_dSet[train_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset.iloc[train_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Converting  data to numpy array
    x_test = np.array(x_test)

    # Reshaping data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the predicted price values from the model
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    # rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    # Plot the data
    train = data[:train_len]
    valid = data[train_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,8))
    plt.title(symbol+' Predictions By LSTM')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(symbol +' Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training the ML-Model', 'Actual Close Price', 'Predicted Close Price'], loc='lower right')
    #plt.show()
    st.pyplot()
    # Show the valid and predicted prices
    st.write(valid)
    plt.figure(figsize=(12,6))
    plt.title(symbol + ' Predictions By LSTM')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Actual Close Price', 'Predicted Close Price'], loc='lower right')
    st.pyplot()
    
    # prepairing the Error Metrics
    resultsdf=  resultsdf.append(my_dict(symbol+':','LSTM:','R-Square Error:',metrics.r2_score(valid['Close'] , valid['Predictions'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','LSTM:','Mean Absolute Error:', metrics.mean_absolute_error(valid['Close'] , valid['Predictions'])),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','LSTM:','Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(valid['Close'] , valid['Predictions']))),ignore_index=True)
    resultsdf = resultsdf.append(my_dict(symbol+':','LSTM:','Mean Squared Error:', metrics.mean_squared_error(valid['Close'] , valid['Predictions'])),ignore_index=True)
       
    st.write(resultsdf)
    
