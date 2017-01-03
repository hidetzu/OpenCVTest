#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

class MaskMethod {
public:
	inline void SetImage(cv::Mat src) { _src = src; };
	virtual cv::Mat Create() = 0;
protected:
	cv::Mat _src;
};
