Title: The Hedge Algorithm on U.S. Technology Stocks
Date: 2018-06-24 8:00 a.m.
Category: Our Projects
Tags: python, machine learning, finance, investing
Authors: A.J. Imholte
Summary: An implementation of the hedge machine learning algorithm in the U.S. stock market.

## Background

The Hedge algorithm was originally presented by Yoav Freund and Robert E. Schapire in 1996 named "_A Decision-Theoretic Generalization of On-Line Learning<br>and an Application to Boosting_" and was presented as a way to solve what is known by those in the field as an _online allocation problem_.

## The Problem

Freund and Schapire (1996) present an anecdotal example of what problem this algorithm seeks to solve in their paper:

> A gambler, frustrated by persistent horse-racing losses
> and envious of his friends' winnings, decides to allow a
> group of his fellow gamblers to make bets on his behalf. He
> decides he will wager a fixed sum of money in every race, but
> that he will apportion his money among his friends based on
> how well they are doing. Certainly, if he knew psychically
> ahead of time which of his friends would win the most, he
> would naturally have that friend handle all his wagers.
> Lacking such clairvoyance, however, he attempts to allocate
> each race's wager in such a way that his total winnings for the season will be reasonably close to what he would have
> won had he bet everything with the luckiest of his friends.

More formally, the problem is defined as follows:

An "allocation agent" (labeled as _A_) has a selection of _N_ possble actions or _strategies_ in each time period _T_, where _N_ and _T_ are both positive constants. In a given time period _t_ from _t = 1,2,3...T_, the agent _A_ selects an "allocation" - or weight - for each one of the possible options. The agent _A_ then observes all the losses that results from playing each strategy with its corresponding weight. Strategies are then reweighted based off these observed losses.

## Pseudocode

The pseudocode for the algorithm is relatively straightforward and is as follows:<br>

![enter image description here](https://lh3.googleusercontent.com/_0y898-oBZjeXn_OFYRRRXU066BhGLyXX_QxgFQOxIgF-IYopak4pwSOYy4WP6czW1oBaPYGOUli)<br>

As you can see, the actual pseudocode for the algorithm is relatively straightforward. Which begs the question - how can we use this in the stock market?

## Rationale for Use in the Stock Market

In a way, the problem is not too different from what many financial institutions must do: create a portfolio of "winning" stocks, where higher weights are used with stocks that are overall better in quality, both from a risk and return standpoint. One problem that many people have with creating their own portfolio is how much of their total portfolio should be comprised of a certain stock. That's where this algorithm comes in. The hedge algorithm will actually _compute_ all the weights for the stocks we are considering in our portfolio, so we don't have to worry ourselves with picking the weight for each individual stock. That sure sounds like it would make life a little easier, right?

## Our Approach

With all this in mind, we wanted to implement the hedge algorithm on a list of stocks from the iShares large cap technology sector ETF to see if the algorithm would outperform the market and the actual ETF itself. In order to do this, we obtained all the necessary stock data and processed it (this will be covered in another blog post). After mining and cleaning our data, we were left with closing prices for each stock in the ETF from 2010 to 2018. With this in hand, we were then able to implement the algorithm in Python using some helpful libraries.

Before I go further, I will note that we tested four different versions of the hedge algorithm for this specific context, which were the following:

1. A *naive*, *unoptimized* hedge algorithm. Basically, the barebones algorithm...with no additional information or tuning.
2. A *naive*, *optimized* algorithm where we still have no additional information to add, but we do test and implement an optimal value for the value of the learning rate, *beta*.
3. An *informed*, *unoptimized* hedge algorithm. This is very similar to the first case, but this time we give the algorithm the weights from the iShares large cap technology ETF as a starting point for all the stocks in consideration. In this way, we *inform* the algorithm with some knowledge of the dynamics of the system it's working with, which hopefully will lead to better results!
4. An *informed*, *optimized* hedge algorithm. This is the cream of the crop as far as our three implementations go, since we are both *adding* additional information (like we do in the third case), but we are also *optimizing* by using an optimal learning rate for this specific situation (like the second case). Due to this, we are hoping this one will have the best results out of the four.


## Our Results

I decided to put this first since the code itself is pretty lengthy. More on that below. In the meantime - I'll talk about the results that we got from this experiment here!

|                            | Sharpe Ratio | Calmar Ratio | Upside Potential | Maximum Drawdown | Volatility | Average Daily Return |
|----------------------------|--------------|--------------|------------------|------------------|------------|----------------------|
| S&P 500                    | 1.62         | 0.50         | 56.0             | 0.96             | 0.29       | 0.48                 |
| iShares ETF                | 1.41         | 0.40         | 34.6             | 1.30             | 0.37       | 0.53                 |
| Unoptimized Naive Hedge    | 1.29         | 0.82         | 158.1            | 0.93             | 0.59       | 0.76                 |
| Optimized Naive Hedge      | 1.34         | 0.58         | 94.7             | 1.19             | 0.52       | 0.70                 |
| Unoptimized Informed Hedge | 1.70         | 0.52         | 177.0            | 1.64             | 0.50       | 0.86                 |
| Optimized Informed Hedge   | 1.52         | 0.44         | 93.2             | 1.60             | 0.46       | 0.70                 |

That's a lot of numbers to take in, so let's talk a little bit more about what this means. Let's also keep in mind our two basic thoughts we had about this experiment as it relates to the stock market, and this algorithm in particular:

1. *More* and *better* information will help us make better returns on our investment.
2. Parameter tuning will lead to algorithm *optimization*, which should improve results.

Let's tackle that first one. **Do we see better returns coming from a program with more information at its disposal?**

For the answer to this question, we can compare the unoptimized naive and informed algorithms from the table above. 

A quick aside here - keep in mind that we want the Sharpe ratio, Calmar ratio, upside potential, and average daily return to be high - whereas we want volatility and maximum drawdown to be low. Think of the Sharpe, Calmar, upside potential, and average daily return as positive feedback that things are working. Conversely, volatility and maximum drawdown represent negative feedback...that things aren't working.

Looking at the performance metrics, our unoptimized informed algorithm beats its counterpart naive algorithm in every category, other than maximum drawdown and Calmar ratio. Looking at our other available benchmarks, the S&P 500's and iShares' performance over this time period, it looks like the informed algorithm is also at least staying on par with its benchmarks. Based off this, I think it's safe to say that our family of informed algorithms are better than our naive algorithms in terms of performance on this dataset. This conclusion confirms our first theory! One down, one to go!

Now that we have the answer to our first question in hand, what about the second one? **Is parameter tuning effective?**

Let's point our attention to optimized/unoptimized algorithms for the answer. Based off the chart, it looks like in both the naive and informed case that it's pretty unclear which is actually performing better. Maybe some graphs can help us out here? <br>

![]({attach}images/informedhedge_vs_stocks.png)<br>

Here we have the unoptimized informed hedge algorithm against all the individual stock returns that we considered in this period. This is a pretty interesting graph. As you can see, the algorithm is not the best performer in group. It's pretty consistently average. <br>
![]({attach}images/informedhedge_vs_market.png)<br>
Now, when we look at it at the market level, all of a sudden its performance looks much better! How come? Because it's **very difficult if not impossible** for this algorithm to consistently beat the best stocks in sector over time, **but** it is possible to outperform the market. This doesn't mean we should start using this for our retirements quite yet and quit our day jobs...more  on this later.<br>

![]({attach}images/informedhedgeoptimized_vs_stocks.png)<br>
Looks pretty similar to what we had before, right?
![]({attach}images/informedhedgeoptimized_vs_market.png)<br>
Compared to the market, that seems to look substantially better than the unoptimized version.

Just to make sure we are on the right track - let's plot the performance of each version of the algorithm against each other: <br>
![MyImage]({attach}images/Figure_1.png)<br>
This actually looks pretty close! Assuming we are using a buy-and-hold strategy, it looks like the *informed*, *optimized* hedge algorithm just barely beats out its optimized, naive counterpart. Looks like our second hypothesis is confirmed!

## The Code

Now, the moment you've all been waiting for...the code itself!

So, this is a pretty big file, about 350 lines of code to be exact! So I'll break down some of the code here and then leave you to explore as you wish.

### Lines 2-8

The first thing that I would like to highlight here are the packages we are importing: RiskAdjustedReturnMetrics, numpy, numpy.linalg, math, csv, matplotlib, and pandas. Let's highlight what these all do:

+ RiskedAdjustedReturnMetrics is a great Github [repository](https://gist.github.com/StuartGordonReid/67a1ec4fbc8a84c0e856#file-riskadjustedreturnmetrics-py]) written by Stuart Gordon Reid from [turingfinance.com](http://www.turingfinance.com/), a great website on computational finance that came in very handy for this project. This repo comes packed with a bunch of useful functions used to diagnose portfolio performance, such as computing Sharpe ratios, maximum drawdown, volatility, and many more. This much-needed package of functions helped  us evaluate the performance of our algorithm. A big thanks to Stuart for this one!

+ Numpy and numpy.linalg are two fairly standard data science Python packages. They give us some nice capabilities for handling the data, along with some useful functions for linear algebra that we will need to optimize operations on our dataset.

+ Math and CSV are both pretty standard when handling data as well. They provided us with some more nice functions for dealing with data and reading our CSV files for our dataset. 

+ Matplotlib and Pandas - also pretty standard. Matplotlib for plotting, Pandas for dealing with data.

As you can see, a lot of these packages are for dealing with data, which might seem a little overkill. But, all these packages put together give us some pretty powerful capabilities that vanilla Python does not, which will make implementing the algorithm much easier. 

<script src="https://gist.github.com/aimholte/b04ecbb37d84b333dc9175759b630779.js"></script>

### Lines 13-40

Here we are doing some more basic setup - reading in our datasets, performing some basic computations on our dataset to get the maximum returns for each day (lines 25-27), constants for the number of stocks we are considering (line 29) and the number of time periods (line 30). Lines 33-39 we initialize what we called the "naive" weights for the stocks.  What this means is that in the first period, all weights for each stock is equal. So if there are about 100 stocks in our dataset, each stock would get a weight of 1% as a starting point. We also provide a value for *beta*, which is the "learning rate" of our algorithm (line 39).  Luckily for us - Freund and Schapire (1996) give us a get starting point to work with from their work.

We also created a list that represented the *informed* weight for each stock. What does that mean exactly? As you'd probably expect, some stocks should definitely have a higher starting weight in our initial portfolio than others, right? There is no reason why a company like Apple, who has consistently outperformed the market for decades, should have the same weight in our portfolio as a small cap tech company that no one has ever heard of. Sure, that rinky-dink, small tech company has the *potential* to grow like  Apple has, but that's not for us to say or decide. Instead, what we are testing here is the idea that *more information is better*. If we know more about how something works, that should help, right?

<script src="https://gist.github.com/aimholte/44593e3471916f2970f98dc2a6f83f61.js"></script>

### Lines 43-98

Here we define the functions *plotProfits()*, *plotPrices()*, *plotChanges()*, and *plotChanges()*. As you can probably guess - these are our main visualization functions. We will use these to chart the algorithm's performance over time.


<script src="https://gist.github.com/aimholte/60741d8b627629cf4c7ab3a6d69acb23.js"></script>


### Lines 101-135

Hey, look! You've made it to *one* of our actual implementations of the hedge algorithm. The function's pretty simple, like it's pseudocode. Most important thing to note here - to determine our results from the period, we have to get the dot product of our stock weights, by the period's current prices, to get our return for the period (line 116). 

After that's done - we determine which stock had the greatest increase and then *penalize every other stock based off how different its change was from this maximum* (lines 120-121).

Once everything is done - we plot our results (lines 132-135).

<script src="https://gist.github.com/aimholte/e17ff17743c59929dfec6e08a01cea0f.js"></script>

### Lines 137-194

This function *hedgeWithReturns()* is the same thing as above with a slight tweak: we *reinvest* our returns from the period back into our portfolio. Warren Buffet would be so proud, right? 

We use some of the functions from RiskAdjustedReturnMetrics here as well: the Sharpe ratio, upside potential, Calmar ratio, volatility, maximum drawdown, and average daily return. For definitions on these, [Stuart's post](http://www.turingfinance.com/computational-investing-with-python-week-one/]) on Turing Finance is a great resource. All you really need to know though is that we want the Sharpe ratio, Calmar ratio, upside potential, and average daily return to be high. Whereas we want volatility and maxiumum drawdown to be low.

<script src="https://gist.github.com/aimholte/b6e86eb45cc24612e9405606f858e2a5.js"></script>


### Lines 196-226

This is basically the same function as above without the graphs. Used this for some testing.

<script src="https://gist.github.com/aimholte/86fba022bff66cde6d37216b8c5b08ce.js"></script>

### Lines 228-248

Another useful graphing function that we made. This one lets us see our returns from the hedge algorithm compared to that of the market and the iShares ETF.

<script src="https://gist.github.com/aimholte/110e0e16d4dbaac74734a0c3c33d03d7.js"></script>

### Lines 250-289

Here we define a pretty straightforward function, *marketDiagnostics()*. All we are really doing here is calling all those functions from the RiskAdjustedMetrics package for the returns generated from that of the Hedge algorithm and those of the S&P 500. This function helped us with our performance evaluation.

<script src="https://gist.github.com/aimholte/d29aa825e9f319626f417c30e08232ff.js"></script>

### Lines 292-305

This is a relatively simple function as well, which we named  *bestBeta*. This one helps us pick an optimal value for our "tuning parameter," the learning rate *beta*. All this function does is loop over a bunch of small, incremental potential values for our learning parameter, determines which one of those values leads to the highest Sharpe ratio for the hedge algorithm, and then spits it back out to us. Simple but effective!

<script src="https://gist.github.com/aimholte/c67f7d14e94449695ce001a812954b1c.js"></script>


**We are getting close to being done, only a few more to go!**

### Lines 307-324
This is yet another graphing function, *plotComparisonsWithMarket()*. The function takes all our different results from all the different versions of the algorithm we tried, graphs them, and displays their performance against that of the market.

<script src="https://gist.github.com/aimholte/4b1096048a6f116a9ffa677ed73fb6e2.js"></script>

### Lines 326-344
Another comparison function - *compareHedgeAlgorithms()*, which compares the performance of each of the versions we tried to one another.

<script src="https://gist.github.com/aimholte/726a3f25abad8dd744cf529a3254bc07.js"></script>

### Lines 346-349
A simple wrapper function which calls *marketDiagnostics()* and *compareHedgeAlgorithms()* that we discussed above.

<script src="https://gist.github.com/aimholte/7044554cab23090e9f31511a4c683704.js"></script>


That's it for the code!

## Critiques
So, we've done all this work, so everything's good right? We can quit our jobs and let this thing make money for us right? Sit back, sleep in, and let this earn money for us? Not quite.

As you can probably guess, trading with just one algorithm - and a simple one at that - is extremely risky. Furthermore, when you are in the market, you are competing directly with large, institutional investors. They have more people than you, more information than you, and more resources than you. Also, people build much more complex models to manage investing than this one. Based on all of this, would you tie your money to something this rudimentary? Probably not. 

That being said - our efforts aren't worthless. This algorithm, and others like it, can be used as a building block for more effective strategies, or as a program to consult with when an investor is considering a move in their portfolio. This gives a lot of power to the investor to make a more educated decision about their investments. I'll take that as a win in my book for sure!


----------------------------------------------------------------------------


That draws an end to our first project post! We are hoping to have many more like this in the future. You can find the full source code for this project at [this Github repository](https://github.com/aimholte/algorithms-project).

**References:**
http://algo.cs.uni-frankfurt.de/~pnakhe/Blog-Hedge/blog.html  
http://www.turingfinance.com/computational-investing-with-python-week-one/
https://gist.github.com/StuartGordonReid/67a1ec4fbc8a84c0e856#file-riskadjustedreturnmetrics-py 
http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf 
http://www.morningstar.com/
https://finance.yahoo.com/
https://www.ishares.com/us
https://pythonprogramming.net/handling-stock-data-graphing-python-programming-for-finance/?completed=/



