My main goal in this educational endeavor is to be able to use the [MLPACK](http://www.mlpack.org/), [Shark](http://image.diku.dk/shark/), and [dlib](http://dlib.net/) C++ machine learning libraries in R for computationally intensive problems. Now, there is a [RcppMLPACK](https://cran.r-project.org/package=RcppMLPACK), but that one apparently uses version 1 of MLPACK (which is now in version 2) and doesn't include any supervised learning methods, just unsupervised learning methods.

-   [Setup](#setup)
    -   [Software Libraries](#software-libraries)
    -   [Mac OS X](#mac-os-x)
    -   [Ubuntu/Debian](#ubuntudebian)
        -   [Building Shark library](#building-shark-library)
    -   [R Packages](#r-packages)
-   [Rcpp](#rcpp)
    -   [Basics](#basics)
    -   [Using Libraries](#using-libraries)
        -   [Armadillo vs RcppArmadillo](#armadillo-vs-rcpparmadillo)
        -   [Fast K-Means](#fast-k-means)
    -   [Fast Classification](#fast-classification)
        -   [External Pointers](#external-pointers)
    -   [Object Serialization](#object-serialization)
        -   [Simple Example](#simple-example)
        -   [Fast Classification Revisited](#fast-classification-revisited)
            -   [Training](#training)
            -   [Prediction](#prediction)
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
brew install mlpack shark dlib
```

Ubuntu/Debian
-------------

``` bash
sudo apt-get install libmlpack-dev libdlib-dev
```

### Building Shark library

If `sudo apt-get install libshark-dev` is no go, we have to build the library ourselves. See [these installation instructions](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/getting_started/installation.html) for more details.

``` bash
# Install dependencies
sudo apt-get install cmake cmake-curses-gui libatlas-base-dev wget

# Download and unpack
wget -O Shark-3.0.0.tar.gz https://github.com/Shark-ML/Shark/archive/v3.0.0.tar.gz
tar -zxvf Shark-3.0.0.tar.gz
mv Shark-3.0.0 Shark

# Configure and build
mkdir Shark/build/
cd Shark/build
# cmake "-DENABLE_OPENMP=OFF" "-DCMAKE_INSTALL_PREFIX=/usr/local/" ../
cmake ../
make
make install
```

R Packages
----------

``` r
install.packages(c("BH", "Rcpp", "RcppArmadillo", "microbenchmark", "devtools"))
devtools::install_github("yihui/printr")
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
library(knitr)
library(printr)
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
| native |  0.0025|  0.0029|  0.0052|  0.0043|  0.0064|  0.0172|    100|
| loop   |  0.9133|  1.0851|  1.3737|  1.2399|  1.6022|  2.7706|    100|
| Rcpp   |  0.0044|  0.0057|  0.0131|  0.0112|  0.0172|  0.0516|    100|

Using Libraries
---------------

### Armadillo vs RcppArmadillo

Use the **depends** attribute to bring in [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo), which is an Rcpp integration of the templated linear algebra library [Armadillo](http://arma.sourceforge.net/). The code below is an example of a fast linear model from Dirk Eddelbuettel.

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

| expr    |     min|      lq|    mean|  median|      uq|     max|  neval|
|:--------|-------:|-------:|-------:|-------:|-------:|-------:|------:|
| lm      |  0.8646|  0.9417|  1.2413|  1.0756|  1.3839|  3.0538|    100|
| fastLm  |  0.0956|  0.1081|  0.1786|  0.1237|  0.1677|  3.4207|    100|
| RcppArm |  0.1202|  0.1394|  0.1830|  0.1557|  0.2106|  0.4561|    100|

### Fast K-Means

Unfortunately, [RcppMLPACK](https://cran.r-project.org/package=RcppMLPACK) uses version 1 of [MLPACK](http://www.mlpack.org/) (now in version 2) and only makes the unsupervised learning methods accessible. (Supervised methods would require returning a trained classifier object to R, which is actually a really difficult problem.)

Okay, let's try to get a fast version of <span title="Bradley, P. S., &amp; Fayyad, U. M. (1998). Refining Initial Points for K-Means Clustering. Icml." style="font-weight: bold;">k-means</span>.

First, install the MLPACK library (see [§ Software Libraries](#software-libraries)), then:

``` r
# Thanks to Kevin Ushey for suggesting Rcpp plugins (e.g. Rcpp:::.plugins$openmp)
registerPlugin("mlpack11", function() {
  return(list(env = list(
    USE_CXX1X = "yes",
    CXX1XSTD="-std=c++11",
    PKG_LIBS = "-lmlpack"
  )))
})
```

We refer to the documentation for [KMeans](http://www.mlpack.org/docs/mlpack-1.0.6/doxygen.php?doc=kmtutorial.html#kmeans_kmtut) shows, although it seems to incorrectly use `arma::Col<size_t>` for cluster assigments while in practice the cluster assignments are returned as an `arma::Row<size_t>` object.

``` cpp
// [[Rcpp::plugins(mlpack11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace Rcpp;
#include <mlpack/core/util/log.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
using namespace mlpack::kmeans;
using namespace arma;

// [[Rcpp::export]]
NumericVector fastKm(const arma::mat& data, const size_t& clusters) {
  Row<size_t> assignments;
  KMeans<> k;
  k.Cluster(data, clusters, assignments);
  // Let's change the format of the output to be a little nicer:
  NumericVector results(data.n_cols);
  for (int i = 0; i < assignments.n_cols; i++) {
    results[i] = assignments(i) + 1; // cluster assignments are 0-based
  }
  return results;
}
```

(Alternatively: `sourceCpp("src/fastKm.cpp")` which creates `fastKm()` from [src/fastKm.cpp](src/fastKm.cpp))

``` r
data(trees, package = "datasets"); data(faithful, package = "datasets")
# KMeans in MLPACK requires observations to be in columns, not rows:
ttrees <- t(trees); tfaithful <- t(faithful)
microbenchmark(
  kmeans_trees = kmeans(trees, 3),
  fastKm_trees = fastKm(ttrees, 3),
  kmeans_faithful = kmeans(faithful, 2),
  fastKm_faithful = fastKm(tfaithful, 2)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr             |     min|      lq|    mean|  median|      uq|     max|  neval|
|:-----------------|-------:|-------:|-------:|-------:|-------:|-------:|------:|
| kmeans\_trees    |  0.2260|  0.2421|  0.3179|  0.2593|  0.3645|  0.8182|    100|
| fastKm\_trees    |  0.0184|  0.0317|  0.0675|  0.0398|  0.0512|  2.5202|    100|
| kmeans\_faithful |  0.2742|  0.2968|  0.3890|  0.3170|  0.4035|  1.9142|    100|
| fastKm\_faithful |  0.0764|  0.1222|  0.1473|  0.1353|  0.1497|  0.4749|    100|

Fast Classification
-------------------

In this exercise, we will train a [Naive Bayes classifier from MLPACK](http://www.mlpack.org/docs/mlpack-2.1.0/doxygen.php?doc=classmlpack_1_1naive__bayes_1_1NaiveBayesClassifier.html). First, we train and classify in a single step. Then we will store the trained classifier in memory, and then later we will be able to save the model. Storing the trained model requires [serialization](https://en.wikipedia.org/wiki/Serialization), the topic of the next section.

``` cpp
// [[Rcpp::plugins(mlpack11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace Rcpp;
#include <mlpack/core/util/log.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
using namespace mlpack::naive_bayes;
using namespace arma;

// [[Rcpp::export]]
NumericVector fastNBC(const arma::mat& training_data, const arma::Row<size_t>& labels, const size_t& classes, const arma::mat& new_data) {
  // Initialization & training:
  NaiveBayesClassifier<> nbc(training_data, labels, classes);
  // Prediction:
  arma::Row<size_t> predictions;
  nbc.Classify(new_data, predictions);
  // Let's change the format of the output to be a little nicer:
  NumericVector results(predictions.n_cols);
  for (int i = 0; i < predictions.n_cols; ++i) {
    results[i] = predictions(i);
  }
  return results;
}
```

For the classification example, we'll use the Iris dataset. (Of course.)

``` r
data(iris, package = "datasets")
set.seed(0)
training_idx <- sample.int(nrow(iris), 0.8 * nrow(iris), replace = FALSE)
training_x <- unname(as.matrix(iris[training_idx, 1:4]))
training_y <- unname(iris$Species[training_idx])
testing_x <- unname(as.matrix(iris[-training_idx, 1:4]))
testing_y <- unname(iris$Species[-training_idx])
# For fastNBC:
ttraining_x <- t(training_x)
ttraining_y <- matrix(as.numeric(training_y) - 1, nrow = 1)
classes <- length(levels(training_y))
ttesting_x <- t(testing_x)
ttesting_y <- matrix(as.numeric(testing_y) - 1, nrow = 1)
```

I kept getting "Mat::col(): index out of bounds" error when trying to compile. I debugged the heck out of it until I finally looked in **naive\_bayes\_classifier\_impl.hpp** and saw:

``` cpp
for (size_t j = 0; j < data.n_cols; ++j)
{
  const size_t label = labels[j];
  ++probabilities[label];
  
  arma::vec delta = data.col(j) - means.col(label);
  means.col(label) += delta / probabilities[label];
  variances.col(label) += delta % (data.col(j) - means.col(label));
}
```

Hence why we run into a problem when we use `as.numeric(training_y)` in R and turn that factor into 1s, 2s, and 3s. This makes sense in retrospect but would have been nice to explicitly know that MLPACK expects training data class labels to be 0-based.

``` r
# Naive Bayes via e1071
naive_bayes <- e1071::naiveBayes(training_x, training_y)
predictions <- e1071:::predict.naiveBayes(naive_bayes, testing_x, type = "class")
confusion_matrix <- caret::confusionMatrix(
  data = predictions,
  reference = testing_y
)
confusion_matrix$table
```

| Prediction/Reference |  setosa|  versicolor|  virginica|
|:---------------------|-------:|-----------:|----------:|
| setosa               |       9|           0|          0|
| versicolor           |       0|          11|          1|
| virginica            |       0|           0|          9|

``` r
print(confusion_matrix$overall["Accuracy"])
```

    ## Accuracy 
    ##   0.9667

``` r
# Naive Bayes via MLPACK
predictions <- fastNBC(ttraining_x, ttraining_y, classes, ttesting_x)
confusion_matrix <- caret::confusionMatrix(
  data = predictions,
  reference = ttesting_y
)
confusion_matrix$table
```

| Prediction/Reference |    0|    1|    2|
|:---------------------|----:|----:|----:|
| 0                    |    9|    0|    0|
| 1                    |    0|   11|    1|
| 2                    |    0|    0|    9|

``` r
print(confusion_matrix$overall["Accuracy"])
```

    ## Accuracy 
    ##   0.9667

``` r
# Performance Comparison
microbenchmark(
  naiveBayes = {
    naive_bayes <- e1071::naiveBayes(training_x, training_y)
    predictions <- e1071:::predict.naiveBayes(naive_bayes, testing_x, type = "class")
  },
  fastNBC = fastNBC(ttraining_x, ttraining_y, classes, ttesting_x)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr       |     min|      lq|    mean|  median|     uq|      max|  neval|
|:-----------|-------:|-------:|-------:|-------:|------:|--------:|------:|
| naiveBayes |  4.5837|  4.8541|  5.9210|  5.3583|  6.558|  10.4261|    100|
| fastNBC    |  0.0152|  0.0174|  0.0372|  0.0399|  0.045|   0.2771|    100|

### External Pointers

In the next step, we'll train a Naive Bayes classifier and keep that trained object in memory to make classification a separate step. Notice that we have to:

-   declare a pointer: `NaiveBayesClassifier<>* nbc = new NaiveBayesClassifier<>(...)`
-   use Rcpp's external pointers (`Rcpp::XPtr`) and
-   return an [S-expression](http://adv-r.had.co.nz/C-interface.html#c-data-structures) (`SEXP`).

``` cpp
// [[Rcpp::plugins(mlpack11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace Rcpp;
#include <mlpack/core/util/log.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
using namespace mlpack::naive_bayes;
using namespace arma;

// [[Rcpp::export]]
SEXP nbTrain(const arma::mat& training_data, const arma::Row<size_t>& labels, const size_t& classes) {
  // Initialization & training:
  NaiveBayesClassifier<>* nbc = new NaiveBayesClassifier<>(training_data, labels, classes);
  Rcpp::XPtr<NaiveBayesClassifier<>> p(nbc, true);
  return p;
}
```

``` r
fit <- nbTrain(ttraining_x, ttraining_y, classes)
str(fit)
```

    ## <externalptr>

`fit` is an external pointer to some memory. When we pass it to a C++ function, it's passed as an R data type (SEXP) that we have to convert to an external pointer before we can use the object's methods. Notice that we're now calling `nbc->Classify()` instead of `nbc.Classify()`.

``` cpp
// [[Rcpp::plugins(mlpack11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace Rcpp;
#include <mlpack/core/util/log.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
using namespace mlpack::naive_bayes;
using namespace arma;

// [[Rcpp::export]]
NumericVector nbClassify(SEXP xp, const arma::mat& new_data) {
  XPtr<NaiveBayesClassifier<>> nbc(xp);
  // Prediction:
  arma::Row<size_t> predictions;
  nbc->Classify(new_data, predictions);
  // Let's change the format of the output to be a little nicer:
  NumericVector results(predictions.n_cols);
  for (int i = 0; i < predictions.n_cols; ++i) {
    results[i] = predictions(i);
  }
  return results;
}
```

``` r
fit_e1071 <- e1071::naiveBayes(training_x, training_y)
# Performance Comparison
microbenchmark(
  `e1071 prediction` = e1071:::predict.naiveBayes(fit_e1071, testing_x, type = "class"),
  `MLPACK prediction` = nbClassify(fit, ttesting_x)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr              |     min|      lq|    mean|  median|      uq|    max|  neval|
|:------------------|-------:|-------:|-------:|-------:|-------:|------:|------:|
| e1071 prediction  |  3.5034|  3.7712|  4.5391|  4.1819|  4.9229|  8.601|    100|
| MLPACK prediction |  0.0095|  0.0111|  0.0541|  0.0284|  0.0354|  2.750|    100|

See [Exposing C++ functions and classes with Rcpp modules](http://dirk.eddelbuettel.com/code/rcpp/Rcpp-modules.pdf) for more information.

Object Serialization
--------------------

Serialization and deserialization require C++11 (`// [[Rcpp::plugins(cpp11)]]`), [cereal](https://github.com/USCiLab/cereal) via [Rcereal](https://cran.rstudio.com/package=Rcereal) (`// [[Rcpp::depends(Rcereal)]]`), and [boost](http://www.boost.org/) via [BH](https://cran.rstudio.com/package=BH) (`// [[Rcpp::depends(BH)]]`). See [Serialization of trie objects](https://github.com/Ironholds/triebeard/issues/9) discussion from Oliver et al.'s [triebeard](https://cran.rstudio.com/package=triebeard) package and [Serialize and Deserialize a C++ Object in Rcpp](http://gallery.rcpp.org/articles/rcpp-serialization/) article.

### Simple Example

### Fast Classification Revisited

#### Training

#### Prediction

References
==========

-   Eddelbuettel, D. (2013). Seamless R and C++ Integration with Rcpp. New York, NY: Springer Science & Business Media. <http://doi.org/10.1007/978-1-4614-6868-4>
-   Wickham, H. A. (2014). Advanced R. Chapman and Hall/CRC. <http://doi.org/10.1201/b17487>
