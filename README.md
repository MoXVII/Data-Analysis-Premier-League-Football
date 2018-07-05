# Data Analysis: Premier League Football

This project makes use of elements from Big Data and Predictive Analytics with the assistance of Python and a variety of libraries such as Pandas,MatPlotLib and Numpy to read data from a CSV file, clean the data of null or stale entries such that exploratory analysis can be conducted and the data can be understood. From this, predictive models can be produced allowing to simulate the outcome of pitting two football teams against eachother based on their individual performance throughout the season 17/18. 


## Built With

* [Python 3] - Base Language.
* [Jupyter Notebook](http://jupyter.org) - Online web editor used to produce Python code along with comments in markdown for ease of understanding and clarity.
* [Numpy](http://www.numpy.org) - Python Package allowing for scientific computing using arrays and usage of random number generators.
* [Pandas](https://pandas.pydata.org) - Library used for data analysis using dataframes as data structures and reading data from CSV input files.
* [MatPlotLib](https://matplotlib.org) - Python plotting library allowing for visualisation of data in histograms, bar charts and graphs.
* [SKLearn](http://scikit-learn.org/stable/) - Machine Learning library in python allowing for logistic and linear regression and model generation.
* [Sublime Text 3](https://www.sublimetext.com) - Text editor used alongside Jupyter Notebook through duration of project.


##Task1_sol.py
* Features reading of input CSV files into a dataframe using Pandas.
* Cleaning of data to remove entries with no existent infromation or null values
* Describing the data which displays relevant information such as number of entries,mean, standard deviations and quartiles.
* Generation of histogram showing information related to specific football teams' performance to be evaluated.
* Using the information gathered, simulate the outcome of pitting two football teams together as a poisson distribution
* Printing the outcome of simulating 500 games between both teams to the user.

##Task2_sol.py
* Producing the required dataframes after reading input CSV files using Pandas.
* Cleaning and normalising data
* Calculating a 5 day moving average in order to make predictions using the model.
* Splitting the data into training and test data where a three quarters of the data is used to train the data.
* Using SKLearn to select the most relevant features required to make predictions with the model.
* Generating the model and attempting to make predictions based on the input parameters and outputting important information such as Y-Axis Intercept, Coefficients and R Squared to the user.
* Plotting final results as a graph.

All this information can be viewed in both python files and the PDF report outlining the findings.