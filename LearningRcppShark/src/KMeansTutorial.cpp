#include <Rcpp.h>
#include <shark/Algorithms/KMeans.h> //k-means algorithm
#include <shark/Models/Clustering/Centroids.h>//model performing hard clustering of points
#include <shark/Models/Clustering/HardClusteringModel.h>//model performing hard clustering of points
#include "utils.h" // some conversion helpers

using namespace shark;
using namespace std;
using namespace Rcpp;

// [[Rcpp::depends(BH)]]
// [[Rcpp::export]]
List SharkKMeansTrain (NumericMatrix X, ssize_t k) {
	// convert data
	UnlabeledData<RealVector> data = NumericMatrixToUnlabeledData(X);
	// std::size_t elements = data.numberOfElements();
	// compute centroids using k-means clustering
	Centroids centroids;
	kMeans (data, k, centroids);
	// convert cluster centers/centroids
	Data<RealVector> const& c = centroids.centroids();
	NumericMatrix cM = DataRealVectorToNumericMatrix (c);
	// cluster training data we are given and convert to Rcpp object
	HardClusteringModel<RealVector> model(&centroids);
	Data<unsigned> clusters = model(data);
	NumericVector labels = LabelsToNumericVector (clusters);
	// return solution found
	Rcpp::List rl = R_NilValue;
	rl = List::create(_["labels"] = labels,
                   _["centroids"] = cM );
	return (rl);
}

// [[Rcpp::depends(BH)]]
// [[Rcpp::export]]
NumericVector SharkKMeansPredict (NumericMatrix X, NumericMatrix centroids) {
	// convert data
	UnlabeledData<RealVector> data = NumericMatrixToUnlabeledData (X);
	// std::size_t elements = data.numberOfElements();
	// convert centroids
	Centroids c (NumericMatrixToDataRealVector (centroids));
	// cluster data we are given and convert to Rcpp object
	HardClusteringModel<RealVector> model (&c);
	Data<unsigned> clusters = model (data);
	NumericVector labels = LabelsToNumericVector (clusters);
	return labels;
}
