#include "ThresholdMethod.h"

cv::Mat ThresholdMethod::Create()
{
	cv::Mat res;
	cv::threshold(this->_src, res, _thresh, 255, CV_THRESH_BINARY);
	return res;
}
