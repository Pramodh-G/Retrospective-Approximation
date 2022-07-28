# Retrospective-Approximation

## Explanation of what is in the notebook folder:

- `datagen.ipynb` is the file used to generate data for the logit model
- `datagen-LR.ipynb` is the notebook used to generate data for a logistic regression experiment
- `datagen-pasupathy` is the notebook used to generate data to reproduce the pasupathy paper.
	- The paper uses a linear regression dataset with 1000 covariates and 10000 dataset size, it is large
	- Hence, I use a dataset with 10000 examples and 300 covariates.
	- I have not included the dataset since it is 300 MB and hence cant fit on github. you can use this to generate the data for yourself.
- `Optimization-Pasupathy-linreg.ipynb` has an example which solves the problem using optim.jl
- `Optimization-Pasupathy-lbfgs-fast.ipynb` uses a version of lbfgs which is optimized and allocates less.
	- it approximates the inverse hessian instead of the actual hessian
	- The idea for memory optimization(to use circular arrays for the gradient and iterate differences) was imported from optim.jl
	- The implementation details were used from wikipedia. The links are provided in the notebook.

## how to run the code

- Clone the directory using `git clone`
- startup the julia REPL. go to package mode and execute `activate` and `instantiate`
- do the following in the repl mode
```julia
using IJulia
notebook(dir="./")
```
- this requires jupyter-notebook to be installed and to be available on path variable.


## A concise report of the work I did

- RA with SGD as the inner solver can beat standard SGD given we hand tune the batch sizes and epsilons.
- This might get tiring for bigger problems, might take up a lot of time.
- I tried to reproduce Pasupathy et.al's paper, which used a custom LBFGS solver.
- Using Optim.jl's implementation of LBFGS did not work when comparing against standard LBFGS because the information from inner iterations was not carried over.
- I implemented a custom LBFGS algorithm which carried over the iterates from one inner iteration to the next.
- I also implemented the stopping tests, which helped remove the effort of tuning epsilons in the inner iterations.
- When LBFGS + stopping tests was used, it was observed that RA + LBFGS performed comparable to LBFGS with stochastic gradients.
