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
    -   [Example](#example)
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
| native |  0.0025|  0.0028|  0.0047|  0.0041|  0.0052|  0.0298|    100|
| loop   |  0.8758|  0.9859|  1.2629|  1.1102|  1.3949|  2.8459|    100|
| Rcpp   |  0.0042|  0.0053|  0.0112|  0.0095|  0.0140|  0.0696|    100|

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
| lm      |  0.8654|  0.9387|  1.2709|  1.0643|  1.4281|  4.5091|    100|
| fastLm  |  0.0940|  0.1151|  0.1671|  0.1383|  0.2059|  0.7095|    100|
| RcppArm |  0.1190|  0.1522|  0.2033|  0.1781|  0.2504|  0.4193|    100|

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
| kmeans\_trees    |  0.2299|  0.2514|  0.3324|  0.2768|  0.3459|  2.5481|    100|
| fastKm\_trees    |  0.0195|  0.0331|  0.0446|  0.0423|  0.0528|  0.0874|    100|
| kmeans\_faithful |  0.2822|  0.3069|  0.3905|  0.3382|  0.4266|  1.2566|    100|
| fastKm\_faithful |  0.0781|  0.1216|  0.1457|  0.1345|  0.1528|  0.2686|    100|

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
| naiveBayes |  4.5974|  5.0490|  6.2130|  5.5430|  6.932|  12.9500|    100|
| fastNBC    |  0.0152|  0.0179|  0.0352|  0.0403|  0.044|   0.0835|    100|

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
SEXP nbTrainXPtr(const arma::mat& training_data, const arma::Row<size_t>& labels, const size_t& classes) {
  // Initialization & training:
  NaiveBayesClassifier<>* nbc = new NaiveBayesClassifier<>(training_data, labels, classes);
  Rcpp::XPtr<NaiveBayesClassifier<>> p(nbc, true);
  return p;
}
```

``` r
fit <- nbTrainXPtr(ttraining_x, ttraining_y, classes)
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
NumericVector nbClassifyXPtr(SEXP xp, const arma::mat& new_data) {
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
  `MLPACK prediction` = nbClassifyXPtr(fit, ttesting_x)
) %>% summary(unit = "ms") %>% knitr::kable(format = "markdown")
```

| expr              |     min|      lq|    mean|  median|     uq|     max|  neval|
|:------------------|-------:|-------:|-------:|-------:|------:|-------:|------:|
| e1071 prediction  |  3.6011|  3.9952|  4.7748|   4.390|  5.322|  9.1300|    100|
| MLPACK prediction |  0.0095|  0.0118|  0.0316|   0.032|  0.038|  0.1407|    100|

See [Exposing C++ functions and classes with Rcpp modules](http://dirk.eddelbuettel.com/code/rcpp/Rcpp-modules.pdf) for more information.

Object Serialization
--------------------

Example
-------

Taking the GPS example from Boost documentation's [tutorial on serialization](http://www.boost.org/doc/libs/1_62_0/libs/serialization/doc/tutorial.html), using some of the code from [Rcereal's README](https://github.com/wush978/Rcereal/blob/master/README.md#getting-started).

``` cpp
// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::depends(BH)]]
#include <boost/archive/binary_oarchive.hpp>

class gps_position
{
public:
    int degrees;
    int minutes;
    float seconds;
    gps_position(){};
    gps_position(int d, int m, float s) :
        degrees(d), minutes(m), seconds(s)
    {}
};

namespace boost {
  namespace serialization {
  
    template<class Archive>
    void serialize(Archive & ar, gps_position & g, const unsigned int version)
    {
        ar & g.degrees;
        ar & g.minutes;
        ar & g.seconds;
    }
  
  } // namespace serialization
} // namespace boost

// [[Rcpp::export]]
RawVector serializeGPS(int d, int m, float s) {
  gps_position g(d, m, s);
  std::stringstream ss;
  {
    boost::archive::binary_oarchive oa(ss);
    oa << g;
  }
  ss.seekg(0, ss.end);
  RawVector retval(ss.tellg());
  ss.seekg(0, ss.beg);
  ss.read(reinterpret_cast<char*>(&retval[0]), retval.size());
  return retval;
}
```

``` r
serializeGPS(35L, 52L, 0.3)
```

    *** caught segfault ***
    address 0x0, cause 'unknown'

AWESOME.

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

Unfortunately, it seems there's a typo in **naive\_bayes\_classifier\_impl.hpp**:

``` cpp
template<typename MatType>
template<typename Archive>
void NaiveBayesClassifier<MatType>::Serialize(Archive& ar,
                                              const unsigned int /* version */)
{
  ar & data::CreateNVP(means, "means");
  ar & data::CreateNVP(variances, "variances");
  ar & data::CreateNVP(probabilities, "probabilities");
}
```

and in **naive\_bayes\_classifier.hpp**:

``` cpp
//! Serialize the classifier.
template<typename Archive>
void Serialize(Archive& ar, const unsigned int /* version */);
```

Namely, that `version` is commented out, meaning that `int` is now seen as a `const unsigned` variable name. IDK, maybe that actually means something, though??? So we have to use what *cereal* refers to as external serialization functions (see [Types of Serialization Functions](http://uscilab.github.io/cereal/serialization_functions.html#types-of-serialization-functions) documentation).

``` cpp
// [[Rcpp::plugins(mlpack11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace Rcpp;

#include <mlpack/core/util/log.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
using namespace mlpack::naive_bayes;

// Serialization:
#define CEREAL_SERIALIZE_FUNCTION_NAME Serialize
// [[Rcpp::depends(Rcereal)]]
#include <sstream>
#include <cereal/archives/binary.hpp>
// #include <mlpack/core/data/serialization_shim.hpp>

// [[Rcpp::export]]
RawVector nbTrain(const arma::mat& training_data, const arma::Row<size_t>& labels, const size_t& classes) {
  // Initialization & training:
  NaiveBayesClassifier<> nbc(training_data, labels, classes);
  // Serialize:
  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
    nbc.Serialize(oarchive);
    // oarchive(nbc);
  }
  ss.seekg(0, ss.end);
  RawVector retval(ss.tellg());
  ss.seekg(0, ss.beg);
  ss.read(reinterpret_cast<char*>(&retval[0]), retval.size());
  return retval;
}
```

``` r
fit <- nbTrain(ttraining_x, ttraining_y, classes)
str(fit)
```

#### Prediction (Deserialization)

...

``` cpp
// [[Rcpp::export]]
NumericVector nbClassify(RawVector src) {
  ...
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

References
==========

-   Eddelbuettel, D. (2013). Seamless R and C++ Integration with Rcpp. New York, NY: Springer Science & Business Media. <http://doi.org/10.1007/978-1-4614-6868-4>
-   Wickham, H. A. (2014). Advanced R. Chapman and Hall/CRC. <http://doi.org/10.1201/b17487>
