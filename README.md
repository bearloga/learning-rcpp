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
    -   [Other Libraries](#other-libraries)
        -   [Shark](#shark)
            -   [Classification](#classification)
        -   [DLib](#dlib)
    -   [Object Serialization](#object-serialization)
        -   [Fast Classification Revisited](#fast-classification-revisited)
            -   [Training (Serialization)](#training-serialization)
            -   [Prediction (Deserialization)](#prediction-deserialization)
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
install.packages(c(
  "BH", # Header files for 'Boost' C++ library
  "Rcpp", # R and C++ integration
  "RcppArmadillo", # Rcpp integration for 'Armadillo' linear algebra library
  "Rcereal", # header files of 'cereal', a C++11 library for serialization
  "microbenchmark", # For benchmarking performance
  "devtools", # For installing packages from GitHub
  "magrittr", # For piping
  "knitr" # For printing tables & data.frames as Markdown
), repos = "https://cran.rstudio.com/")
devtools::install_github("yihui/printr") # Prettier table printing
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
library(BH)
library(Rcpp)
library(RcppArmadillo)
library(Rcereal)
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
| native |  0.0026|  0.0033|  0.0057|  0.0045|  0.0064|  0.0375|    100|
| loop   |  0.8924|  1.1060|  1.5682|  1.2689|  1.7722|  6.0524|    100|
| Rcpp   |  0.0047|  0.0071|  0.0154|  0.0126|  0.0171|  0.1171|    100|

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

| expr    |     min|      lq|    mean|  median|      uq|    max|  neval|
|:--------|-------:|-------:|-------:|-------:|-------:|------:|------:|
| lm      |  0.8616|  1.0288|  1.6195|  1.3594|  1.9909|  4.793|    100|
| fastLm  |  0.0936|  0.1216|  0.2150|  0.1608|  0.2310|  1.236|    100|
| RcppArm |  0.1195|  0.1502|  0.2574|  0.2063|  0.3216|  1.640|    100|

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
NumericVector mlpackKM(const arma::mat& data, const size_t& clusters) {
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
# kmeans coerces data frames to matrix, so it's worth doing that beforehand
mtrees <- as.matrix(trees)
mfaithful <- as.matrix(faithful)
# KMeans in MLPACK requires observations to be in columns, not rows:
ttrees <- t(trees); tfaithful <- t(faithful)
microbenchmark(
  kmeans_trees = kmeans(mtrees, 3),
  mlpackKM_trees = mlpackKM(ttrees, 3),
  kmeans_faithful = kmeans(mfaithful, 2),
  mlpackKM_faithful = mlpackKM(tfaithful, 2)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr               |     min|      lq|    mean|  median|      uq|      max|  neval|
|:-------------------|-------:|-------:|-------:|-------:|-------:|--------:|------:|
| kmeans\_trees      |  0.1862|  0.2095|  0.4912|  0.2663|  0.4899|   8.0418|    100|
| mlpackKM\_trees    |  0.0175|  0.0367|  0.0636|  0.0508|  0.0809|   0.2474|    100|
| kmeans\_faithful   |  0.1981|  0.2198|  0.4983|  0.2993|  0.5202|  10.1142|    100|
| mlpackKM\_faithful |  0.0793|  0.1258|  0.2050|  0.1462|  0.2576|   1.3892|    100|

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

// [[Rcpp::export]]
NumericVector mlpackNBC(const arma::mat& training_data, const arma::Row<size_t>& labels, const size_t& classes, const arma::mat& new_data) {
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
predictions <- mlpackNBC(ttraining_x, ttraining_y, classes, ttesting_x)
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
  fastNBC = mlpackNBC(ttraining_x, ttraining_y, classes, ttesting_x)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr       |     min|      lq|    mean|  median|      uq|      max|  neval|
|:-----------|-------:|-------:|-------:|-------:|-------:|--------:|------:|
| naiveBayes |  4.5083|  4.9263|  5.7663|  5.3907|  6.1432|  10.3640|    100|
| fastNBC    |  0.0149|  0.0173|  0.0336|  0.0369|  0.0422|   0.1244|    100|

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

// [[Rcpp::export]]
SEXP mlpackNBTrainXPtr(const arma::mat& training_data, const arma::Row<size_t>& labels, const size_t& classes) {
  // Initialization & training:
  NaiveBayesClassifier<>* nbc = new NaiveBayesClassifier<>(training_data, labels, classes);
  Rcpp::XPtr<NaiveBayesClassifier<>> p(nbc, true);
  return p;
}
```

``` r
fit <- mlpackNBTrainXPtr(ttraining_x, ttraining_y, classes)
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

// [[Rcpp::export]]
NumericVector mlpackNBClassifyXPtr(SEXP xp, const arma::mat& new_data) {
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
  `MLPACK prediction` = mlpackNBClassifyXPtr(fit, ttesting_x)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr              |     min|      lq|    mean|  median|     uq|     max|  neval|
|:------------------|-------:|-------:|-------:|-------:|------:|-------:|------:|
| e1071 prediction  |  3.5286|  3.8864|  4.6389|  4.4660|  5.000|  8.0984|    100|
| MLPACK prediction |  0.0093|  0.0112|  0.0257|  0.0265|  0.035|  0.0854|    100|

See [Exposing C++ functions and classes with Rcpp modules](http://dirk.eddelbuettel.com/code/rcpp/Rcpp-modules.pdf) for more information.

Other Libraries
---------------

### Shark

For this one, I'm putting my work/notes into the [LearningRcppShark](LearningRcppShark/) package, which is how I'm learning the [Shark library](http://image.diku.dk/shark/), creating R bindings to it via RcppShark, and learning how to make an R package that makes use of an Rcpp-based library wrapper.

``` r
# devtools::install_github("bearloga/learning-rcpp/LearningRcppShark")
library(LearningRcppShark)
```

``` r
fit <- shark_kmeans(training_x, 3)
predictions <- predict(fit, testing_x)
```

``` r
# This version does not check if x is a numeric matrix
sharkKM <- function(x, k) {
  return(LearningRcppShark:::SharkKMeansTrain(x, k))
}

microbenchmark(
  kmeans_trees = kmeans(mtrees, 3),
  mlpackKM_trees = mlpackKM(ttrees, 3),
  shark_km_trees = shark_kmeans(mtrees, 3),
  sharkKM_trees = sharkKM(mtrees, 3),
  kmeans_faithful = kmeans(mfaithful, 2),
  mlpackKM_faithful = mlpackKM(tfaithful, 2),
  shark_km_faithful = shark_kmeans(mfaithful, 2),
  sharkKM_faithful = sharkKM(mfaithful, 2)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr                |     min|      lq|    mean|  median|      uq|      max|  neval|
|:--------------------|-------:|-------:|-------:|-------:|-------:|--------:|------:|
| kmeans\_trees       |  0.1963|  0.2252|  0.3561|  0.2765|  0.3858|   2.1254|    100|
| mlpackKM\_trees     |  0.0229|  0.0369|  0.0582|  0.0478|  0.0686|   0.2531|    100|
| shark\_km\_trees    |  0.1015|  0.1284|  0.1987|  0.1536|  0.2299|   0.7095|    100|
| sharkKM\_trees      |  0.0658|  0.0871|  0.1253|  0.1020|  0.1322|   0.5512|    100|
| kmeans\_faithful    |  0.2109|  0.2455|  0.3604|  0.3010|  0.4372|   0.8352|    100|
| mlpackKM\_faithful  |  0.0787|  0.1241|  0.1549|  0.1406|  0.1624|   0.3529|    100|
| shark\_km\_faithful |  0.2831|  0.3367|  0.6385|  0.3690|  0.5502|  12.1438|    100|
| sharkKM\_faithful   |  0.2576|  0.2949|  0.3952|  0.3198|  0.4443|   1.1611|    100|

#### Classification

### DLib

``` r
registerPlugin("dlib11", function() {
  return(list(env = list(
    USE_CXX1X = "yes",
    CXX1XSTD="-std=c++11",
    PKG_LIBS = "-ldlib"
  )))
})
```

``` cpp
// [[Rcpp::plugins(dlib11)]]
#include <Rcpp.h>
using namespace Rcpp;

#include <shark/Algorithms/KMeans.h> // k-means algorithm
#include <shark/Models/Clustering/HardClusteringModel.h>// model performing hard clustering of points
```

Object Serialization
--------------------

``` cpp
// [[Rcpp::plugins(cpp11)]]

/* The serialization/deserialization function name to search for.
   You can define CEREAL_SERIALIZE_FUNCTION_NAME to be different,
   assuming you do so before <cereal/macros.hpp> is included. */
#define CEREAL_SERIALIZE_FUNCTION_NAME Serialize
// [[Rcpp::depends(Rcereal)]]

#include <sstream>
#include <cereal/archives/binary.hpp>
#include <Rcpp.h>

struct MyClass
{
  int x, y, z;

  // This method lets cereal know which data members to serialize
  template<class Archive>
  void Serialize(Archive& archive)
  {
    archive( x, y, z ); // serialize things by passing them to the archive
  }
};

using namespace Rcpp;
//[[Rcpp::export]]
RawVector serialize_myclass(int x = 1, int y = 2, int z = 3) {
  MyClass my_instance;
  my_instance.x = x;
  my_instance.y = y;
  my_instance.z = z;
  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
    oarchive(my_instance);
  }
  ss.seekg(0, ss.end);
  RawVector retval(ss.tellg());
  ss.seekg(0, ss.beg);
  ss.read(reinterpret_cast<char*>(&retval[0]), retval.size());
  return retval;
}

//[[Rcpp::export]]
void deserialize_myclass(RawVector src) {
  std::stringstream ss;
  ss.write(reinterpret_cast<char*>(&src[0]), src.size());
  ss.seekg(0, ss.beg);
  MyClass my_instance;
  {
    cereal::BinaryInputArchive iarchive(ss);
    iarchive(my_instance);
  }
  Rcout << my_instance.x << "," << my_instance.y << "," << my_instance.z << std::endl;
}
```

``` r
raw_vector <- serialize_myclass(1, 2, 4)
str(raw_vector)
deserialize_myclass(raw_vector)
```

### Fast Classification Revisited

#### Training (Serialization)

#### Prediction (Deserialization)

References
==========

-   Eddelbuettel, D. (2013). Seamless R and C++ Integration with Rcpp. New York, NY: Springer Science & Business Media. <http://doi.org/10.1007/978-1-4614-6868-4>
-   Wickham, H. A. (2014). Advanced R. Chapman and Hall/CRC. <http://doi.org/10.1201/b17487>
