#pragma once

#include "MaskMethod.h"

class ThresholdMethod : public MaskMethod {
public:
	ThresholdMethod(double thresh) { _thresh = thresh; };
	cv::Mat Create();

private:
	double _thresh;
};
