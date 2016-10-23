My main goal in this educational endeavor is to be able to use the [MLPACK](http://www.mlpack.org/) library in R, with the hope of being able to include its Naive Bayes classifier in [MLPUGS](https://cran.r-project.org/package=MLPUGS). Now, there is a [RcppMLPACK](https://cran.r-project.org/package=RcppMLPACK), but that one apparently uses version 1 of MLPACK (which is now in version 2) and doesn't include any supervised learning methods, just unsupervised learning methods.

-   [Setup](#setup)
    -   [Software Libraries](#software-libraries)
    -   [Mac OS X](#mac-os-x)
    -   [Ubuntu/Debian](#ubuntudebian)
    -   [R Packages](#r-packages)
-   [Rcpp](#rcpp)
    -   [Basics](#basics)
    -   [Using Libraries](#using-libraries)
    -   [Modules](#modules)
-   [References](#references)

Setup
=====

Software Libraries
------------------

Mac OS X
--------

``` bash
## To install Homebrew:
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
## Then:
brew tap homebrew/versions && brew tap homebrew/science && brew update
# brew install gcc --enable-cxx && brew link --overwrite gcc && brew link cmake
brew install boost --c++11
# Installing cmake may require: sudo chown -R `whoami` /usr/local/share/man/man7
brew install mlpack
```

Ubuntu/Debian
-------------

``` bash
sudo apt-get install libmlpack
```

R Packages
----------

``` r
install.packages(c("BH", "Rcpp", "RcppArmadillo", "microbenchmark"))
```

If you get "ld: library not found for -lgfortran" error when trying to install RcppArmadillo, run:

``` bash
curl -O http://r.research.att.com/libs/gfortran-4.8.2-darwin13.tar.bz2
sudo tar fvxz gfortran-4.8.2-darwin13.tar.bz2 -C /
```

See "[Rcpp, RcppArmadillo and OS X Mavericks "-lgfortran" and "-lquadmath" error](http://thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/)"" for more info.

Rcpp
====

See [this section](http://rmarkdown.rstudio.com/authoring_knitr_engines.html#rcpp) in [RMarkdown documentation](http://rmarkdown.rstudio.com/) for details on Rcpp chunks.

``` r
library(magrittr)
library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)
```

Basics
------

``` cpp
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector cumSum(NumericVector x) {
  int n = x.size();
  NumericVector out(n);
  out[0] = x[0];
  for (int i = 1; i < n; ++i) {
    out[i] = out[i-1] + x[i];
  }
  return out;
}
```

``` r
x <- 1:1000
microbenchmark(
  native = cumsum(x),
  loop = (function(x) {
    output <- numeric(length(x))
    output[1] <- x[1]
    for (i in 2:length(x)) {
      output[i] <- output[i-1] + x[i]
    }
    return(output)
  })(x),
  Rcpp = cumSum(x)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr   |     min|      lq|    mean|  median|      uq|     max|  neval|
|:-------|-------:|-------:|-------:|-------:|-------:|-------:|------:|
| native |  0.0025|  0.0027|  0.0041|  0.0031|  0.0043|  0.0285|    100|
| loop   |  0.8592|  0.9358|  1.1537|  1.0114|  1.1266|  2.5243|    100|
| Rcpp   |  0.0043|  0.0049|  0.0097|  0.0066|  0.0119|  0.0922|    100|

Using Libraries
---------------

This is the example code taken from Dirk's talk on RcppArmadillo and I'm using it to test how much I can do within Rcpp chunks in RMarkdown.

``` cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::export]]
List fastLm(const colvec& y, const mat& X) {
  int n = X.n_rows, k = X.n_cols;
  colvec coef = solve(X, y);
  colvec resid = y - X*coef;
  double sig2 = as_scalar(trans(resid) * resid/(n-k));
  colvec stderrest = sqrt(sig2 * diagvec( inv(trans(X)*X)) );
  return List::create(_["coefficients"] = coef,
                      _["stderr"]       = stderrest,
                      _["df.residual"]  = n - k );
}
```

``` r
data("mtcars", package = "datasets")
microbenchmark(
  lm = lm(mpg ~ wt + disp + cyl + hp, data = mtcars),
  fastLm = fastLm(mtcars$mpg, cbind(1, as.matrix(mtcars[, c("wt", "disp", "cyl", "hp")]))),
  RcppArm = fastLmPure(cbind(1, as.matrix(mtcars[, c("wt", "disp", "cyl", "hp")])), mtcars$mpg)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr    |     min|      lq|    mean|  median|      uq|    max|  neval|
|:--------|-------:|-------:|-------:|-------:|-------:|------:|------:|
| lm      |  0.8363|  0.8800|  1.0891|  0.9287|  1.2126|  3.148|    100|
| fastLm  |  0.0909|  0.1088|  0.1464|  0.1213|  0.1480|  1.566|    100|
| RcppArm |  0.1153|  0.1295|  0.1736|  0.1453|  0.1708|  1.411|    100|

Okay, let's try to adapt [RcppMLPACK's example code](https://github.com/thirdwing/RcppMLPACK/wiki/Example#k-means-example) for [k-means](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=kmtutorial.html#kmeans_kmtut) to a RcppMLPACK-less context :)

``` cpp
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace Rcpp;
#include <mlpack/methods/kmeans/kmeans.hpp>
using namespace mlpack::kmeans;

// [[Rcpp::export]]
arma::Row<size_t> kMeans(const arma::mat& data, const int& clusters) {
    arma::Row<size_t> assignments;
    KMeans<> k;
    k.Cluster(data, clusters, assignments); 
    return assignments;
}
```

I get the following:

    Building shared library for Rcpp code chunk...

    Quitting from lines 132-147 (README.Rmd) 
    Error in dyn.load("/Users/mpopov/Desktop/Learning Rcpp/README_cache/markdown_github/kmeans_source_sourceCpp/sourceCpp-x86_64-apple-darwin13.4.0-0.12.7/sourcecpp_450b02624e/sourceCpp_2.so") : 
      unable to load shared object '/Users/mpopov/Desktop/Learning Rcpp/README_cache/markdown_github/kmeans_source_sourceCpp/sourceCpp-x86_64-apple-darwin13.4.0-0.12.7/sourcecpp_450b02624e/sourceCpp_2.so':
      dlopen(/Users/mpopov/Desktop/Learning Rcpp/README_cache/markdown_github/kmeans_source_sourceCpp/sourceCpp-x86_64-apple-darwin13.4.0-0.12.7/sourcecpp_450b02624e/sourceCpp_2.so, 6): Symbol not found: __ZN6mlpack3Log4InfoE
      Referenced from: /Users/mpopov/Desktop/Learning Rcpp/README_cache/markdown_github/kmeans_source_sourceCpp/sourceCpp-x86_64-apple-darwin13.4.0-0.12.7/sourcecpp_450b02624e/sourceCpp_2.so
      Expected in: flat namespace
     in /Users/mpopov/Desktop/Learning Rcpp/README_cache/markdown_github/kmeans_source_sourceCpp/sourceCpp-x86_64-apple-darwin13.4.0-0.12.7/sourcecpp_450b02624e/sourceCpp_2.so
    Calls: <Anonymous> ... <Anonymous> -> source -> withVisible -> eval -> eval -> dyn.load
    Execution halted

``` r
data(trees, package = "datasets")
microbenchmark(
  kmeans = kmeans(trees, 3),
  kMeans = kMeans(t(trees), 3)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

Modules
-------

### Simple

References
==========

-   Eddelbuettel, D. (2013). Seamless R and C++ Integration with Rcpp. New York, NY: Springer Science & Business Media. <http://doi.org/10.1007/978-1-4614-6868-4>
