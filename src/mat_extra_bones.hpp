//! Add a serialization operator.
template<typename Archive>
void serialize(Archive& ar, const unsigned int version);

#include <RcppArmadillo/Mat_proto.h>
