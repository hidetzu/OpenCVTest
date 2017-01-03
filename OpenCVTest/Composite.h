#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "MaskMethod.h"

class Composite {
private:
	cv::Mat CreateMaskImg();

public:
	Composite(cv::Mat portraitPhotograph, cv::Mat backgroundPicture, MaskMethod* maskFactory);
	cv::Mat Execute();

private:
	MaskMethod* _maskMethod;
	cv::Mat _portraitPhotograph;
	cv::Mat _backgroundPicture;

	cv::Mat _gray;
	cv::Mat _mask;
};
