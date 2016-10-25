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
  NumericVector clust(data.n_cols);
  KMeans<> k;
  k.Cluster(data, clusters, assignments);
  for (int i = 0; i < assignments.n_cols; i++) {
    clust[i] = assignments(i) + 1; // cluster assignments are 0-based
  }
  return clust;
}
