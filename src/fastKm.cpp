// [[Rcpp::plugins(mlpack11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <mlpack/core/util/log.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

// [[Rcpp::export]]
Rcpp::NumericVector fastKm(const arma::mat& data, const size_t& clusters) {
  arma::Row<size_t> assignments;
  Rcpp::NumericVector clust(data.n_cols);
  mlpack::kmeans::KMeans<> k;
  mlpack::Log::Info << "Obtaining clusters..." << std::endl;
  k.Cluster(data, clusters, assignments);
  // cluster assignments are 0-based
  for (int i = 0; i < assignments.n_cols; i++) {
    clust[i] = assignments(i) + 1;
  }
  return clust;
}
