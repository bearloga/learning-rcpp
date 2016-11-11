#' @title Train a k-means model
#' @param x A \code{matrix} of observations to cluster
#' @param k Number of clusters
#' @examples
#' y <- shark_kmeans(iris[, 1:4], 3)
#' @export
shark_kmeans <- function(x, k) {
  if (!checkmate::test_matrix(x, mode = "numeric")) {
    x <- as.matrix(x)
  }
  if (!checkmate::test_numeric(k)) {
    k <- as.numeric(k)[1]
  }
  model <- SharkKMeansTrain(x, k)
  model$labels <- model$labels + 1
  return(structure(model, class = "SharkKM"))
}

#' @title Predict using a k-means
#' @param x A \code{SharkKM} model object
#' @param newdata A \code{matrix} to classify
#' @param ... Further arguments passed to or from other methods
#' @examples
#' y <- shark_kmeans(iris[, 1:4], 3)
#' y_hat <- predict(y, iris[, 1:4])
#' @export
predict.SharkKM <- function(obj, newdata, ...) {
  if (!checkmate::test_matrix(newdata, mode = "numeric")) {
    newdata <- as.matrix(newdata)
  }
  predictions <- SharkKMeansPredict(newdata, obj$centroids)
  return(predictions + 1)
}
