//
// utils.h
//			this is part of the RcppShark package (http://github.com/aydindemircioglu/RcppShark)
//
// Copyright (C) 2015 		Aydin Demircioglu <aydin.demircioglu/at/ini.rub.de>
//
// This file is part of the RcppShark library for GNU R.
// It is made available under the terms of the GNU General Public
// License, version 2, or at your option, any later version,
// incorporated herein by reference.
//
// This program is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE.  See the GNU General Public License for more
// details.
// 
// Please do not use this software to destroy or spy on people, environment or things.
// All negative use is prohibited.
//

#ifndef utils_h
#define utils_h

using namespace std;

#include <shark/Data/Dataset.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace shark;


//' Convert numeric vector to Data<unsigned int> (aka labels).
//'
//' @param X		NumericVector to convert
//' @param verbose		be verbose?
Data<unsigned int> NumericVectorToLabels (NumericVector X, bool verbose = false);



//' Convert NumericMatrix to UnlabeledData<RealVector>.
//'
//' @param X		NumericMatrix to convert
//' @param verbose		be verbose?
UnlabeledData<RealVector> NumericMatrixToUnlabeledData (NumericMatrix X, bool verbose = false);



//' Convert NumericMatrix to Data<RealVector>.
//'
//' @param X		NumericMatrix to convert
//' @param verbose		be verbose?
Data<RealVector> NumericMatrixToDataRealVector (NumericMatrix X, bool verbose = false);



//' Convert UnlabeledData<RealVector> to NumericMatrix.
//'
//' @param X		UnlabeledData<RealVector> to convert
//' @param verbose		be verbose?
NumericMatrix UnlabeledDataToNumericMatrix (UnlabeledData<RealVector> X, bool verbose = false);



//' Convert Data<RealVector> to NumericMatrix.
//'
//' @param X		NumericVector to convert
//' @param verbose		be verbose?
NumericMatrix DataRealVectorToNumericMatrix (Data<RealVector> X, bool verbose = false);



//' Convert NumericVector to Data<RealVector>.
//'
//' @param X		NumericVector to convert
//' @param verbose		be verbose?
Data<RealVector> NumericVectorToDataRealVector (NumericVector X, bool verbose = false);



//' Convert Data<unsigned int> (aka labels) to NumericVector.
//'
//' @param X		NumericVector to convert
//' @param verbose		be verbose?
NumericVector LabelsToNumericVector (Data<unsigned int> X, bool verbose = false);



//' Convert NumericVector to RealVector.
//'
//' @param X		NumericVector to convert
//' @param verbose		be verbose?
RealVector NumericVectorToRealVector (NumericVector X, bool verbose = false);



//' Convert RealVector to NumericVector.
//'
//' @param X		NumericVector to convert
//' @param verbose		be verbose?
NumericVector RealVectorToNumericVector (RealVector X, bool verbose = false);
	

#endif
