// OpenCVTest.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "MaskMethod.h"
#include "ThresholdMethod.h"
#include "Composite.h"

int main(int argc, const char* argv[])
{
	//�摜�̓ǂݍ���
	cv::Mat img_src1 = cv::imread("sora.jpg", cv::IMREAD_COLOR);
	cv::Mat img_src2 = cv::imread("dambo3.jpg", cv::IMREAD_COLOR);
	if (img_src1.empty()) 
		return -1;

	if (img_src2.empty())
		return -1;

	// TODO: ��l���̕��@�͌Œ�臒l�@�𗘗p����B
	auto maskFactory = new ThresholdMethod(245);
	Composite* composite = new Composite(img_src2, img_src1, maskFactory);
	cv::Mat result = composite->Execute();

	// ���ʂ̕\��
	cv::imshow("Show MASK COMPOSITION Image", result);
	delete composite;
	composite = nullptr;

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}