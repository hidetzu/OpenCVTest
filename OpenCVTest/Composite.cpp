#include "Composite.h"

Composite::Composite(cv::Mat portraitPhotograph, cv::Mat backgroundPicture, MaskMethod* maskMethod)
{
	_portraitPhotograph = portraitPhotograph;
	_backgroundPicture = backgroundPicture;
	_maskMethod = maskMethod;
};

cv::Mat Composite::Execute()
{
	cv::Mat res;

	//�O���[�X�P�[����
	cv::cvtColor(_portraitPhotograph, _gray, cv::COLOR_BGR2GRAY);

	_maskMethod->SetImage(_gray);
	_mask = _maskMethod->Create();
	cv::imwrite("output_binary.jpg", _mask);


	// �l����𒊏o
	cv::Mat portraitPhotographTmp;
	{
		cv::Mat notImg;
		cv::bitwise_not(_mask, notImg);
		_portraitPhotograph.copyTo(portraitPhotographTmp, notImg);
		cv::imwrite("portraitPhotographTmp.jpg", portraitPhotographTmp);
	}

	// �w�i��𒊏o 
	cv::Mat backgroundPictureTmp;
	{
		_backgroundPicture.copyTo(backgroundPictureTmp, _mask);
		cv::imwrite("backgroundPictureTmp.jpg", backgroundPictureTmp);
	}

	// �l����Ɣw�i�����������B
	cv::bitwise_or(backgroundPictureTmp, portraitPhotographTmp, res);

	return res;
}
