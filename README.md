### Mussard & Pi Alperin (2020), "Accounting for risk factors on health outcomes: The case of Luxembourg", _European Journal of Operational Research_, [open acess click here](https://doi.org/10.1016/j.ejor.2020.09.040)


The python code "A-curves.py" allows to replicate similar achievement curves as those presented in the paper from Mussard & Pi Alperin (2020). "Accounting for risk factors on health outcomes: The case of Luxembourg".

In particular, this python code allows drowing achievement curves accounting for:

(1) the risk sensibility of the social planner (parameter alpha in [0,200])

(2) the redistributive principles (parameter nu in {2,3,4})

(3) confidence intervals (level in [0.01,0.1])

(4) different weighting schemes for the aggregation of health dimensions (Cerioli and Zani (1990), Betti and Verma (1998), and Equal weight)


## Data to be prepared for the python code:

(1) text file (no head title in the file)

(2) the first column must contain the rank of the individus in [0,1] on the basis of their equivalent income level (from the poorest 0 to the richest 1: there is no need to sort the data)

(3) the other columns must contain the health dimensions, i.e. the health deprivation of each individual in the sample, a continuous variable in [0,1] increasing with deprivation or a boolean one in {0,1} (with 0 no deprivation and 1 totally deprived)

(4) the last column must contain the risk factor: a continuous variable in [0,1] increasing with deprivation or a boolean one in {0,1} (with 0 no affected by the risk factor and 1 totally affected)

(5) See the files "data_risk_alcohol.txt" and "data_risk_smoke.txt" as examples.


## References
Betti, G. & Verma, V. K. (1998), Measuring the degree of poverty in a dynamic and comparative context: a multi-dimensional approach using fuzzy set theory, Working Paper 22, Dipartimento di Metodi Quantitativi, Università di Siena.

Cerioli, A. and S. Zani (1990), A Fuzzy Approach to the Measurement of Poverty, in Dagum C. and Zenga M. (eds.), Income and Wealth Distribution, Inequality and Poverty, Springer Verlag, Berlin, 272-284.
