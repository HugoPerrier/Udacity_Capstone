Udcity Capstone
===============

# 1. Introduction
## Stock Market
The ownership of a company can belong to a single person but most of the time the ownership is divided between several shareholders. The value of these company shares depend of the total market valuation of the company which may change over time depending on a variety of factors. Stocks or company shares can be bought and sold, there is therefore a market for company shares called the Stock Market. Stock trades are performed in stock exchanges such as :
- NASDAQ
- London Stock Exchange Group
- Tokyo Stock Exchange Group
Depending on the country they are based in, these stock exchanges have different opening days and times. Stock trades can only be executed during the opening hours of a stock exchange. 
The most valuable companies in 2015 according to the FT500 ranking (link) are:
- Apple
- Exxon Mobil
- Berkshire Hathaway
- Google
- Microsoft 

A good capacity to understand and prectict movements in the stock market is crucial for investers to make profitable investements. To help investers choose which investement will be most profitable, they can use large amount of data on the history of the stock prices of companies. To process all these data, predictive models are created using machine learning. Machine learning models are "trained" to predict future stock prices based on a set of available features. This report explores how machine learning can be used to predict stock prices.

## Objective of the project
The objective of this project is to create a predictive tool that uses machine learning to predict future value of the NASDAQ 100 index (NDX) using historical stock prices of different companies. The NASDAQ 100 index is a stock market index related to the capitalization value of the 100 largest non-financial companies [(NASDAQ 100)](http://www.nasdaq.com/markets/indices/nasdaq-100.aspx).

Section 2 of this report describes the data used to create a machine learning model for stock prices prediction, section 3 describes the strategy to build the model and the data preprocessing operations and section 4 shows results of the predictive model built.

## Software requirements
The software requirements to build the predictive models are:
- python: Main programing language
- numpy: Package for scientific computing in python
- pandas: Package for data structures with python 
- matplotlib: Plotting with python
- sklearn: Package for machine learning in python
- urllib: Package for data fetching across the web
- re: Package for regular expression operations
- Quandl: API to access historical stock prices from Quandl databases using Python

# 2. Data description
## Historical stock prices data
To create a machine learning model to make predictions, it is necessary to first "train" the model using past data. In the context of stock market pricing, the model is trained using historical data of the stock prices. For example we can use data from the past period 2003 to 2005 to train a model and then use that model to make predictions about the future. The stock price data consist of the following informations:
| Open | High | Low | Close | Volume | Dividend | Split Ratio |  
|:-----:|:------:|:------:|:-----:|:------:|:------:|:-----:|
and ajusted values:
| Adj. Open | Adj. Close | Adj. Low | Adj. High | Adj. Volume |
|:-----:|:------:|:------:|:-----:|:------:|

For a given day, the "Open" and "Close" values are the values of a stock at the opening and closing of the stock exchange. The "High" and "Low" value are the maximum and minimum values that the stock has reached during that day. The "Volume" is the total anount of stock that were sold on that day. Divident are XXX, Split ratio is XXX and the adjusted data are XXX.


## Data Acquisition
**Quandl API**
The python Quandl API allows users to query historical stock prices from databases. With the free version only one data point per day can be accessed and there is a maximum amount of queries that can be performed in every 24h period. This is used to get the stock prices of the companies in the NASDAQ 100.

**Pandas stock price data reader**
Historical stock prices from yahoo finance can be queried directly using a pandas module. This is used to get the values of the NASDAQ 100 index.

**List of NASDAQ 100 companies**
To obtain the current list of companies included in the NASDAQ 100 index, web scraping is performed on the [NASDAQ website](http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx).

## Data Format
The data obtained from Quandl API queries are in a pandas DataFrame format. The size of a dataset corresponding one year of historical stock prices for a single company is about 26kb in size.

## Data Preprocessing
### Missing data
Historical stock prices for a company on a given day may not be available for the following reasons:
- The company didn't exist at that time (Google was created in 1998)
- The company existed but was not traded in the stock market (Facebook entered the stock market in 2012)
- The stock market where that company's stock is traded was closed on that day
- The historical stock prices of that company are freely accessible in Quandl API

However, to train a machine learning model, the datasets can not contain missing values. Two options are available to ensure the dataset does not contain missing values:
- Model the missing values
- Remove the datapoints containing missing values

Both option can impact the predictions therefore the way missing values are handled needs to be explained.
In this project, all the companies whose data were not accessible through Quandl were not included in the model. If on a given day some companies have stock price data but others don't, the missing data are replaced with the data from the previous working day (This is called a forward fill method). 

### Feature naming
When the data for different companies are queried from Quandl, they all have the same feature names, it is thus necessary to run a renaming operation when joining the datasets of the different companies. The company "ticker" symbol is simply added to the feature name: "Open" feature for "Apple" company (Ticker "AAPL") becomes "AAPL_Open".

### Data clustering
Each of the companies in the NASDAQ 100 has an influence on the value of the NASDAQ 100 index value but using the historical data from all these companies to create a machine learning model would have a high computational cost (100 companies * 270 working day per year per company * 12 values per working day = 324000 values per year). It would then take a long time to train the models on a laptop. 

To reduce the amount of data to work with companies that have similar behaviors can be grouped together. To do so, an unsupervised learning "clustering technique" is used: KMeans clustering. The Kmean clustering method takes as input the historical stock prices of all companies and a user defined number of desired clusters and outputs a list of companies in each cluster. 
In practice, we calculate the daily variation for each datapoint: "Variation" = "Close" - "Open" and use the "Variation" variable as input data for the clustering.

### Split of the data into train, cross validation and test datasets
To create a machine learning model we create a set of data called training set for which the true value of NDX are known. This set is used to find the model (with fixed hyperparameters) coefficients that minimize the error between predictions and true values of NDX. 
Then a cross validation set similar to the training set is used to find the hyperparameters that make the predictions as general as possible. In other words, the model shouldn't just be able to predict NDX values from the dataset used for training but it should  predict well NDX values for any other dataset.
Finally a test set is created to evaluate the performance of the model on data that were not used in the training process. The test dataset has to contain data posterior to the data in the training and cross validation sets as it is not possible to train a model using data from the future.

### Creation of the final features
The objective of a predictive model is to predict future values of the NASDAQ 100 index, therefore company stock prices of day N should be used to predict NDX value of day N+1 or N+x. It is thus necessary to shift in time the NDX column compared to the feature columns.

Finally, we need to choose which data are used as features to predict NDX, we can't use all of the historical data of every company prior to day N+1 to predict NDX on day N+1. It is therefore necessary to decide how many historical data to use for prediction, which features we want to use and adact the dataset accordingly. A mock-up dataset that could be used in a machine learning model using just the "Open" data from the previous 2 days of companies "X" and "Y" is shown below (the date is written down to help with explanation but it is not used as a feature):

| Date | X Open Day N-2 | X Open Day N-1 | Y Open Day N-2 | Y Open Day N-1 | NDX day N |
|:-----:|:-----:|:------:|:------:|:-----:|:------:|
| 01/01/98 | 100 | 105 | 10 | 11 | 99 |
| 02/01/98 | 105 | 110 | 11 | 12 | 87 |

# 3. Results and analysis
## Missing companies in the data acquisition process
Data for some companies couldn't be accessed using Quandl:
- JD, NCLH : Non american companies data are not available in the Quandl WIKI free database
- KHC : Kraft Heinz Company didn't exist in 2013, Kraft and Heinz merged in 2015
- PYPL : Paypal was a wholly owned subsidiary of eBay until 2015
- WBA : Walgreens Boots didn't exist in 2013

## Company clusters
We apply the clustering method described in the previous section for different number of clusters. The data used for the clustering corresponds to the stock prices data of the NASDAQ 100 companies for the year 2013. Results for nCluster = 10 are shown below:

| Cluster number | List of companies |
|:-----:|:-----:|
| 1 | Analog Devices, Inc., Applied Materials, Inc., Broadcom Limited, Intel Corporation, KLA-Tencor Corporation, Lam Research Corporation, Microchip Technology Incorporated, Maxim Integrated Products, Inc., NVIDIA Corporation, QUALCOMM Incorporated, Skyworks Solutions, Inc., Texas Instruments Incorporated, Xilinx, Inc. |
| 2 | Dollar Tree, Inc., Fastenal Company, Hasbro, Inc., Marriott International, Mattel, Inc., Mondelez International, Inc., Ross Stores, Inc., Tractor Supply Company, Ulta Beauty, Inc. |
| 3 | American Airlines Group, Inc., Adobe Systems Incorporated, Akamai Technologies, Inc., Amazon.com, Inc., Baidu, Inc., Facebook, Inc., Liberty Interactive Corporation, Mylan N.V., Netflix, Inc., The Priceline Group Inc., TripAdvisor, Inc., Tesla, Inc., Yahoo! Inc. |
| 4 | Alexion Pharmaceuticals, Inc., Amgen Inc., Biogen Inc., BioMarin Pharmaceutical Inc., Celgene Corporation, Gilead Sciences, Inc., Incyte Corporation, Regeneron Pharmaceuticals, Inc. |
| 5 | Automatic Data Processing, Inc., Cognizant Technology Solutions Corporation, Monster Beverage Corporation, Paychex, Inc. |
| 6 | Charter Communications, Inc., Comcast Corporation, Costco Wholesale Corporation, DISH Network Corporation, Liberty Global plc |
| 7 | Activision Blizzard, Inc, Electronic Arts Inc., Intuitive Surgical, Inc. |
| 8 | Apple Inc., Autodesk, Inc., Cerner Corporation, Cisco Systems, Inc., CSX Corporation, Cintas Corporation, Discovery Communications, Inc., Discovery Communications, Inc., Expedia, Inc., Fiserv, Inc., Twenty-First Century Fox, Inc., Twenty-First Century Fox, Inc., Alphabet Inc., Illumina, Inc., Intuit Inc., J.B. Hunt Transport Services, Inc., Micron Technology, Inc., O'Reilly Automotive, Inc., PACCAR Inc., Starbucks Corporation, Seagate Technology PLC, Symantec Corporation, Viacom Inc., Vodafone Group Plc, Verisk Analytics, Inc., Western Digital Corporation |
| 9 | CA Inc., Check Point Software Technologies Ltd., Citrix Systems, Inc., eBay Inc., Express Scripts Holding Company, Microsoft Corporation, SBA Communications Corporation, Sirius XM Holdings Inc., T-Mobile US, Inc. |
| 10 | Hologic, Inc., Henry Schein, Inc., Vertex Pharmaceuticals Incorporated, DENTSPLY SIRONA Inc. |

In this exemple, we can see that cluster 3 contains mostly tech companies, cluster 4 contains pharma companies and cluster 7 contains video game companies. Some clusters are more difficult to describe such as cluster 2 that contains hotel, food, beauty and variety store companies.

It should be noted that the clustering is not very stable, the following parameters significantly change the content of clusters:
- Time period of the input data 
- Number of desired clusters
- Number of inititalisation (Kmeans clustering methods randomly initialize the center (in parameter space) of the clusters, the initialization might affect the final clustering so several runs with different random initialization are performed)

It is still a good way to reduce the amount of data we use to build the predictive model.

## Next day NDX prediction : Basics
This section describes the general procedure followed to build a predictive model:
1. Clustering
    - All Nasdaq 100 Company 2013 stock price data are acquired
    - Company that can't be accessed with Quandl are removed from the company list
    - Missing values in the dataset are filled (forward fill)
    - The "Variation" = "Close" - "Open" variable is calculated
    - The desired number of company clusters are created using the "Variation" feature as input data.
    - The first company (in alphabetical order) of each cluster is chosen to represent the whole cluster.
    - A list of selected companies is saved to be used in the predictive model creation.
2. Predictive model
    - Stock price data of the companies chosen in the clustering process are acquired. The 2014-2015 period is used for the trainning/cross validation datasets and the 2016 period is used for the test set.
    - Missing values are filled (forward fill)
    - The "Variation" = "Close" - "Open" variable is calculated. (It can be used as a potential engineered feature)
    - NASDAQ 100 index data are acquired from yahoo finance (using pandas finance data reader).
    - The feature to be used to build the predictive model are chosen:
        - Type of stock price data (ex. ["Open", "Close"] or ["Open", "Variation", "Close"])
        - Number of days used for prediction (ex. data from the last 3 days, data from the last 7 days, ...)
    - The final feature matrix is created (drop undesired features, create previous days features, give the feature matrix the right shape, etc)
    - Choose a machine learning regression model
    - Train the model using the training data
    - Predict NDX and score the model on the train, cv and test sets

## Next day NDX prediction : Number of clusters
In this section the number of clusters is varied and we show the influence on the prediction results.
![alt text](Udacity_Capstone/Figures/NDX_Clusters_cv.pdf)



## Blahh
## Blahh
## Blahh
Dillinger is a cloud-enabled, mobile-ready, offline-storage, AngularJS powered HTML5 Markdown editor.

  - Type some Markdown on the left
  - See HTML in the right
  - Magic

You can also:
  - Import and save files from GitHub, Dropbox, Google Drive and One Drive
  - Drag and drop files into Dillinger
  - Export documents as Markdown, HTML and PDF

Markdown is a lightweight markup language based on the formatting conventions that people naturally use in email.  As [John Gruber] writes on the [Markdown site][df1]

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.

This text you see here is *actually* written in Markdown! To get a feel for Markdown's syntax, type some text into the left window and watch the results in the right.

### Tech

Dillinger uses a number of open source projects to work properly:

* [AngularJS] - HTML enhanced for web apps!
* [Ace Editor] - awesome web-based text editor
* [markdown-it] - Markdown parser done right. Fast and easy to extend.
* [Twitter Bootstrap] - great UI boilerplate for modern web apps
* [node.js] - evented I/O for the backend
* [Express] - fast node.js network app framework [@tjholowaychuk]
* [Gulp] - the streaming build system
* [keymaster.js] - awesome keyboard handler lib by [@thomasfuchs]
* [jQuery] - duh

And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

### Installation

Dillinger requires [Node.js](https://nodejs.org/) v4+ to run.

Download and extract the [latest pre-built release](https://github.com/joemccann/dillinger/releases).

Install the dependencies and devDependencies and start the server.

```sh
$ cd dillinger
$ npm install -d
$ node app
```

For production environments...

```sh
$ npm install --production
$ npm run predeploy
$ NODE_ENV=production node app
```

### Plugins

Dillinger is currently extended with the following plugins

* Dropbox
* Github
* Google Drive
* OneDrive

Readmes, how to use them in your own application can be found here:

* [plugins/dropbox/README.md] [PlDb]
* [plugins/github/README.md] [PlGh]
* [plugins/googledrive/README.md] [PlGd]
* [plugins/onedrive/README.md] [PlOd]

### Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantanously see your updates!

Open your favorite Terminal and run these commands.

First Tab:
```sh
$ node app
```

Second Tab:
```sh
$ gulp watch
```

(optional) Third:
```sh
$ karma start
```
#### Building for source
For production release:
```sh
$ gulp build --prod
```
Generating pre-built zip archives for distribution:
```sh
$ gulp build dist --prod
```
### Docker
Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 80, so change this within the Dockerfile if necessary. When ready, simply use the Dockerfile to build the image.

```sh
cd dillinger
npm run-script build-docker
```
This will create the dillinger image and pull in the necessary dependencies. Moreover, this uses a _hack_ to get a more optimized `npm` build by copying the dependencies over and only installing when the `package.json` itself has changed.  Look inside the `package.json` and the `Dockerfile` for more details on how this works.

Once done, run the Docker image and map the port to whatever you wish on your host. In this example, we simply map port 8000 of the host to port 80 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart="always" <youruser>/dillinger:latest
```

Verify the deployment by navigating to your server address in your preferred browser.

```sh
127.0.0.1:8000
```

#### Kubernetes + Google Cloud

See [KUBERNETES.md](https://github.com/joemccann/dillinger/blob/master/KUBERNETES.md)


#### docker-compose.yml

Change the path for the nginx conf mounting path to your full path, not mine!

### N|Solid and NGINX

More details coming soon.


### Todos

 - Write Tests
 - Rethink Github Save
 - Add Code Comments
 - Add Night Mode

License
----

MIT


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [@thomasfuchs]: <http://twitter.com/thomasfuchs>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [keymaster.js]: <https://github.com/madrobby/keymaster>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]:  <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
