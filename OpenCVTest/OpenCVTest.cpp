// OpenCVTest.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "MaskMethod.h"
#include "ThresholdMethod.h"
#include "Composite.h"

#include <vector>
#include <thread>
#include <string.h>
#include <stdlib.h>

#define _WINSOCK_DEPRECATED_NO_WARNINGS 

#include <winsock2.h>

// need link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

void sample1()
{
	//�摜�̓ǂݍ���
	cv::Mat img_src1 = cv::imread("sora.jpg", cv::IMREAD_COLOR);
	cv::Mat img_src2 = cv::imread("dambo3.jpg", cv::IMREAD_COLOR);
	if (img_src1.empty())
		return;

	if (img_src2.empty())
		return;

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
}

void sample3()
{
	//�摜�̓ǂݍ���
	cv::Mat img_src1 = cv::imread("sample_fg.jpg", cv::IMREAD_COLOR);
	cv::Mat img_src2 = cv::imread("sample_bg.jpg", cv::IMREAD_COLOR);
	//cv::Mat img_src2(cv::Size(img_src1.cols, img_src1.rows), CV_8UC3, cv::Scalar::all(255));
	if (img_src1.empty())
		return;

	if (img_src2.empty())
		return;

	cv::Mat hsv;
	cv::cvtColor(img_src1, hsv, cv::COLOR_BGR2HSV);

	cv::Mat sampleImg;
	cv::Canny(img_src1, sampleImg,  50, 110);
	cv::imwrite("Canny.png", sampleImg);
	cv::Mat sampleNotImg;
	cv::bitwise_not(sampleImg, sampleNotImg);
	cv::imwrite("CannyNot.png", sampleNotImg);

	//OpenCV HSV H : 0 - 180, S : 0 - 255, V : 0 - 255
	//	lower_color = np.array([100, 110, 30]) # �F��Ԃ̉���
	//	upper_color = np.array([120, 255, 255]) # �F��Ԃ̏��
	cv::Mat mask;
	cv::inRange(hsv,
		cv::Scalar(40,  120, 10, 0),
		cv::Scalar(100, 255, 255, 0),
		mask);

	cv::imwrite("mask_before.png", mask);
//	cv::bitwise_not(sampleImg, sampleNotImg);
//	cv::bitwise_and(mask, sampleNotImg, mask);

	// �m�C�Y����
#if false
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  cv::Mat(), cv::Point(-1, -1), 1);
#endif

	cv::imwrite("mask.png", mask);

	// �l����𒊏o
	cv::Mat portraitPhotographTmp;
	{
		cv::Mat notImg;
		cv::bitwise_not(mask, notImg);
		img_src1.copyTo(portraitPhotographTmp, notImg);
		cv::imwrite("portraitPhotographTmp.jpg", portraitPhotographTmp);
	}

	// �w�i��𒊏o 
	cv::Mat backgroundPictureTmp;
	{
		img_src2.copyTo(backgroundPictureTmp, mask);
		cv::imwrite("backgroundPictureTmp.jpg", backgroundPictureTmp);
	}

	// �l����Ɣw�i�����������B
	cv::Mat result;
	cv::bitwise_or(backgroundPictureTmp, portraitPhotographTmp, result);

#if false
	cv::morphologyEx(result, result, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
	cv::morphologyEx(result, result, cv::MORPH_OPEN,  cv::Mat(), cv::Point(-1, -1), 1);
#endif
	//cv::imshow("Show MASK COMPOSITION Image", result);
	cv::imwrite("result.jpg", result);

#if false
	cv::Mat notImg;
	cv::Mat result;
	cv::bitwise_not(mask, notImg); // �}�X�N�𔽓]
	cv::imwrite("notmask.png", notImg);
	cv::bitwise_and(img_src1, img_src1, result, notImg);

	// ���ʂ̕\��
	cv::imshow("Show MASK COMPOSITION Image", result);
#endif

	cv::waitKey(0);
	cv::destroyAllWindows();
}

void sample4()
{
	//�摜�̓ǂݍ���
	cv::Mat img_src1 = cv::imread("sample_fg.jpg", cv::IMREAD_COLOR);
	cv::Mat img_src2 = cv::imread("sample_fg_bg.jpg", cv::IMREAD_COLOR);
	cv::Mat img_back = cv::imread("sample_bg.jpg", cv::IMREAD_COLOR);
	//cv::Mat img_src2(cv::Size(img_src1.cols, img_src1.rows), CV_8UC3, cv::Scalar::all(255));
	if (img_src1.empty())
		return;

	if (img_src2.empty())
		return;

	cv::Mat img_src1_gray;
	cv::Mat img_src2_gray;
	cv::cvtColor(img_src1, img_src1_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img_src2, img_src2_gray, cv::COLOR_BGR2GRAY);

	cv::Mat matDiff;
	cv::absdiff(img_src1_gray, img_src2_gray, matDiff);

	cv::morphologyEx(matDiff, matDiff, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
	cv::morphologyEx(matDiff, matDiff, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 1);
	cv::imwrite("matDiff.png", matDiff);

	cv::Mat binaryImg;
	cv::threshold(matDiff, binaryImg, 20, 255, CV_THRESH_BINARY_INV);;
	cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
	cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 1);

	cv::imwrite("matDiff_binary.png", binaryImg);

	// �l����𒊏o
	cv::Mat portraitPhotographTmp;
	{
		cv::Mat notImg;
		cv::bitwise_not(binaryImg, notImg);
		img_src1.copyTo(portraitPhotographTmp, notImg);
		cv::imwrite("portraitPhotographTmp.jpg", portraitPhotographTmp);
	}

	// �w�i��𒊏o 
	cv::Mat backgroundPictureTmp;
	{
		img_back.copyTo(backgroundPictureTmp, binaryImg);
		cv::imwrite("backgroundPictureTmp.jpg", backgroundPictureTmp);
	}

	// �l����Ɣw�i�����������B
	cv::Mat result;
	cv::bitwise_or(backgroundPictureTmp, portraitPhotographTmp, result);
	//cv::imshow("Show MASK COMPOSITION Image", result);
	cv::imwrite("result.jpg", result);
}

void sample5()
{
	//�摜�̓ǂݍ���
	cv::Mat img_src1 = cv::imread("b.png", cv::IMREAD_COLOR);
	cv::Mat img_src2 = cv::imread("a.png", cv::IMREAD_COLOR);
	cv::Mat img_back(cv::Size(img_src1.cols, img_src1.rows), CV_8UC3, cv::Scalar::all(255));
	if (img_src1.empty())
		return;

	if (img_src2.empty())
		return;

	cv::Mat img_src1_gray;
	cv::Mat img_src2_gray;
	cv::cvtColor(img_src1, img_src1_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img_src2, img_src2_gray, cv::COLOR_BGR2GRAY);

	cv::Mat matDiff;
	cv::absdiff(img_src1, img_src2, matDiff);
	cv::cvtColor(matDiff, matDiff, cv::COLOR_BGR2GRAY);

#if false
	cv::morphologyEx(matDiff, matDiff, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
	cv::morphologyEx(matDiff, matDiff, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 1);
#endif
	cv::imwrite("matDiff.png", matDiff);

	cv::Mat binaryImg;
	cv::threshold(matDiff, binaryImg, 1, 255, CV_THRESH_BINARY_INV);
#if true
	cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
	cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 1);
#endif

	cv::imwrite("matDiff_binary.png", binaryImg);

	// �l����𒊏o
	cv::Mat portraitPhotographTmp;
	{
		cv::Mat notImg;
		cv::bitwise_not(binaryImg, notImg);
		img_src1.copyTo(portraitPhotographTmp, notImg);
		cv::imwrite("portraitPhotographTmp.jpg", portraitPhotographTmp);
	}

	// �w�i��𒊏o 
	cv::Mat backgroundPictureTmp;
	{
		img_back.copyTo(backgroundPictureTmp, binaryImg);
		cv::imwrite("backgroundPictureTmp.jpg", backgroundPictureTmp);
	}

	// �l����Ɣw�i�����������B
	cv::Mat result;
	cv::bitwise_or(backgroundPictureTmp, portraitPhotographTmp, result);
	//cv::imshow("Show MASK COMPOSITION Image", result);
	cv::imwrite("result.jpg", result);
}

void soft_mask(
	cv::Mat&  image,
	cv::Mat&  keyImage,
	int thdh, int thdl)
{
	int     d;
	int     kk;
	for (int y = 0; y < image.rows; ++y)
	{
		uchar* imgPtr  = image.ptr<uchar>(y);
		uchar* keyImgPtr = keyImage.ptr<uchar>(y);
		for (int x = 0; x < image.cols; ++x)
		{
			auto imgBgr = imgPtr[x];
			d = imgBgr;
			kk = ((long)(d - thdl) * 255 / (thdh - thdl));
			if (kk > 255)        keyImgPtr[x] = 255;
			else if (kk < 0)     keyImgPtr[x] = 0;
			else                 keyImgPtr[x] = kk;
		}
	}
}

void synth(cv::Mat& fgImg, cv::Mat& bgImg, cv::Mat& keyImg, cv::Mat* pOutImg)
{
	for (int y = 0; y < fgImg.rows; ++y)
	{
		cv::Vec3b* fgImgPtr  = fgImg.ptr<cv::Vec3b>(y);
		cv::Vec3b* bgImgPtr  = bgImg.ptr<cv::Vec3b>(y);
		cv::Vec3b* outImgPtr = pOutImg->ptr<cv::Vec3b>(y);
		uchar* keyImgPtr = keyImg.ptr<uchar>(y);
		for (int x = 0; x < fgImg.cols; ++x)
		{
			auto fgBgr = fgImgPtr[x];
			auto bgBgr = bgImgPtr[x];

			auto rr1 = (int)fgBgr[2];
			auto gg1 = (int)fgBgr[1];
			auto bb1 = (int)fgBgr[0];
			auto rr2 = (int)bgBgr[2];
			auto gg2 = (int)bgBgr[1];
			auto bb2 = (int)bgBgr[0];
			long kk = (long)keyImgPtr[x];

			auto rr = (unsigned char)((rr1*kk + rr2*(255 - kk)) / 255);
			auto gg = (unsigned char)((gg1*kk + gg2*(255 - kk)) / 255);
			auto bb = (unsigned char)((bb1*kk + bb2*(255 - kk)) / 255);

			outImgPtr[x] = cv::Vec3b(bb, gg, rr);
		}
	}
}

void s_synth(cv::Mat& fgImg, cv::Mat& bgImg, cv::Mat& keyImg, cv::Mat* pOutImg)
{
	for (int y = 0; y < fgImg.rows; ++y)
	{
		cv::Vec3b* fgImgPtr  = fgImg.ptr<cv::Vec3b>(y);
		cv::Vec3b* bgImgPtr  = bgImg.ptr<cv::Vec3b>(y);
		cv::Vec3b* outImgPtr = pOutImg->ptr<cv::Vec3b>(y);
		uchar* keyImgPtr = keyImg.ptr<uchar>(y);
		for (int x = 0; x < fgImg.cols; ++x)
		{
			auto fgBgr = fgImgPtr[x];
			auto bgBgr = bgImgPtr[x];

			auto rr1 = (int)fgBgr[2];
			auto gg1 = (int)fgBgr[1];
			auto bb1 = (int)fgBgr[0];
			auto rr2 = (int)bgBgr[2];
			auto gg2 = (int)bgBgr[1];
			auto bb2 = (int)bgBgr[0];
			long kk = (long)keyImgPtr[x];

			if (kk == 255 || kk == 0) {       /* �O�i�܂��͔w�i */
				auto rr = (uchar)((rr1*kk + rr2*(255 - kk)) / 255);
				auto gg = (uchar)((gg1*kk + gg2*(255 - kk)) / 255);
				auto bb = (uchar)((bb1*kk + bb2*(255 - kk)) / 255);
				outImgPtr[x] = cv::Vec3b(bb, gg, rr);
			}
			else {                              /* ���E�� */
				auto rr = (unsigned char)((gg1*kk + rr2*(255 - kk)) / 255);
				auto gg = (unsigned char)((gg1*kk + gg2*(255 - kk)) / 255);
				auto bb = (unsigned char)((gg1*kk + bb2*(255 - kk)) / 255);
				outImgPtr[x] = cv::Vec3b(bb, gg, rr);
			}
		}
	}
}

void bitwise_andEx(cv::Mat& src1, cv::Mat& src2, cv::Mat* pOutImg)
{
}

void sample6()
{
	//�摜�̓ǂݍ���
	cv::Mat fgImg = cv::imread("sample_fg.jpg", cv::IMREAD_COLOR);
	cv::Mat bgImg = cv::imread("sample_fg_bg.jpg", cv::IMREAD_COLOR);
	cv::Mat img_back(cv::Size(fgImg.cols, fgImg.rows), CV_8UC3, cv::Scalar::all(255));
	cv::Mat result = img_back.clone();

	cv::Mat matDiff;
	cv::absdiff(fgImg, bgImg, matDiff);
	cv::cvtColor(matDiff, matDiff, cv::COLOR_BGR2GRAY);
	cv::imwrite("matDiff.png", matDiff);

	cv::Mat binaryImg(cv::Size(fgImg.cols, fgImg.rows), CV_8UC1, cv::Scalar::all(255));
//	soft_mask(matDiff, binaryImg, 20, 40);
//	cv::bitwise_not(binaryImg, binaryImg);
	cv::threshold(matDiff, binaryImg, 40, 255, CV_THRESH_BINARY_INV);
	cv::imwrite("binary.png", binaryImg);

#if false
	s_synth(fgImg, img_back, binaryImg, &result);
#else
	{
		// �l����𒊏o
		cv::Mat portraitPhotographTmp;
		{
			cv::Mat notImg;
			cv::bitwise_not(binaryImg, notImg);
			fgImg.copyTo(portraitPhotographTmp, notImg);
			cv::imwrite("portraitPhotographTmp.jpg", portraitPhotographTmp);
		}

		// �w�i��𒊏o 
		cv::Mat backgroundPictureTmp;
		{
			img_back.copyTo(backgroundPictureTmp, binaryImg);
			cv::imwrite("backgroundPictureTmp.jpg", backgroundPictureTmp);
		}

		// �l����Ɣw�i�����������B
		cv::bitwise_or(backgroundPictureTmp, portraitPhotographTmp, result);
		//cv::imshow("Show MASK COMPOSITION Image", result);
	}
#endif
	cv::imwrite("result.jpg", result);

#if false
	cv::absdiff(result, bgImg, matDiff);
	cv::cvtColor(matDiff, matDiff, cv::COLOR_BGR2GRAY);
	cv::imwrite("matDiff2.png", matDiff);

	cv::Mat binaryImg2(cv::Size(fgImg.cols, fgImg.rows), CV_8UC1, cv::Scalar::all(255));
//	soft_mask(matDiff, binaryImg2, 20, 50);
	cv::threshold(matDiff, binaryImg2, 10, 255, CV_THRESH_BINARY);
//	cv::bitwise_not(binaryImg2, binaryImg2);
	cv::imwrite("binary2.png", binaryImg2);
#endif

	{

		// ���͉摜�̎擾
		cv::Mat im = result;
		cv::Mat im2(im.size(), im.type());

		// �摜�z���1�����ɕϊ�
		cv::Mat points;
		im.convertTo(points, CV_32FC3);
		points = points.reshape(3, im.rows*im.cols);

		// RGB��Ԃ�k-means�N���X�^�����O�����s
		cv::Mat_<int> clusters(points.size(), CV_32SC1);
		cv::Mat centers;

		// �N���X�^��
		const int cluster = 5;
		// k-meansn�N���X�^�����O�̎��s
		kmeans(points, cluster, clusters, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);

		// ��f�l���N���X�^���S�l�ɒu��
		cv::MatIterator_<cv::Vec3b>   itd = im2.begin<cv::Vec3b>();
		cv::MatIterator_<cv::Vec3b>   itd_end = im2.end<cv::Vec3b>();
		for (int i = 0; itd != itd_end; ++itd, ++i) {
			cv::Vec3f &color = centers.at<cv::Vec3f>(clusters(i), 0);
			(*itd)[0] = cv::saturate_cast<uchar>(color[0]);
			(*itd)[1] = cv::saturate_cast<uchar>(color[1]);
			(*itd)[2] = cv::saturate_cast<uchar>(color[2]);
		}
		// ���ʕ\��
		imshow("Input", im);
		imshow("Output", im2);
		cv::absdiff(im, im2, matDiff);
		imshow("Output2", matDiff);
	}

	cv::Mat binaryImg3(cv::Size(fgImg.cols, fgImg.rows), CV_8UC1, cv::Scalar::all(255));
	//	soft_mask(matDiff, binaryImg2, 20, 50);
	cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
	cv::imwrite("result_gray.png", result);
	cv::threshold(result, binaryImg3, 40, 255, CV_THRESH_BINARY);
	cv::imwrite("binary2.png", binaryImg3);

	cv::imwrite("result2.jpg", result);
	cv::waitKey();
}

//���F�p�̊֐�
int convert(int value) {
	if (value < 32) {
		return 32;
	}
	else if (value < 64) {
		return 64;
	}
	else if (value < 96) {
		return 96;
	}
	else if (value < 128) {
		return 128;
	}
	else if (value < 160) {
		return 160;
	}
	else if (value < 192) {
		return 192;
	}
	else if (value < 224) {
		return 224;
	}
	else {
		return 255;
	}
	return 0;  // �����B
}

void sample7()
{
	using namespace cv;
	using namespace std;

	//�摜�̓ǂݍ��݂ƕ\��
	Mat input = imread("sample_fg.jpg", cv::IMREAD_COLOR);
	if (input.empty()) {
		cout << "Failed to load image!" << endl;
		return;
	}
	imshow("original", input);

	//���K���E���F
	Mat norm(input.size(), input.type());
	Mat sample(input.size(), input.type());
	normalize(input, norm, 0, 255, NORM_MINMAX, CV_8UC3);
	for (int y = 0; y<input.rows; y++) {
		for (int x = 0; x<input.cols; x++) {
			for (int c = 0; c < input.channels(); ++c) {
				int index = y*input.step + x*input.elemSize() + c;
				sample.data[index] = convert(norm.data[index]);
			}
		}
	}
	input = sample;
	imshow("down", input);

	//kmeans���s�����߂Ƀt�H�[�}�b�g��ϊ�
	Mat matrix = input.reshape(1, input.cols*input.rows);
	matrix.convertTo(matrix, CV_32FC1, 1.0 / 255.0);

	//kmeans�̎��s
	Mat centers, labels; //���ʂ̊i�[
	TermCriteria criteria(TermCriteria::COUNT, 100, 1); //100��J��Ԃ�
	int cluster_number = 6; //�N���X�^��
	kmeans(matrix, cluster_number, labels, criteria, 1, KMEANS_RANDOM_CENTERS, centers);

	//���C���[�̏���(������Ԃ𔒂œ���)
	Mat layer0(Size(input.cols, input.rows), CV_8UC3, Scalar::all(255));
	Mat layer1(Size(input.cols, input.rows), CV_8UC3, Scalar::all(255));
	Mat layer2(Size(input.cols, input.rows), CV_8UC3, Scalar::all(255));
	Mat layer3(Size(input.cols, input.rows), CV_8UC3, Scalar::all(255));
	Mat layer4(Size(input.cols, input.rows), CV_8UC3, Scalar::all(255));
	Mat layer5(Size(input.cols, input.rows), CV_8UC3, Scalar::all(255));

	//�F�ƃ��x�����̏���
	MatConstIterator_<int> label_first = labels.begin<int>();
	centers.convertTo(centers, CV_8UC1, 255.0);
	centers = centers.reshape(3);

	//���C���[���ƂɐF�𕪂���
	for (int y = 0; y<input.rows; ++y) {
		for (int x = 0; x<input.cols; ++x) {
			const Vec3b& rgb = centers.ptr<Vec3b>(*label_first)[0];
			if (*label_first == 0) layer0.at<Vec3b>(y, x) = rgb;
			else if (*label_first == 1) layer1.at<Vec3b>(y, x) = rgb;
			else if (*label_first == 2) layer2.at<Vec3b>(y, x) = rgb;
			else if (*label_first == 3) layer3.at<Vec3b>(y, x) = rgb;
			else if (*label_first == 4) layer4.at<Vec3b>(y, x) = rgb;
			else layer5.at<Vec3b>(y, x) = rgb;
			++label_first;
		}
	}
	//���C���[���Ƃɕ\��
	imshow("layer0", layer0);
	imshow("layer1", layer1);
	imshow("layer2", layer2);
	imshow("layer3", layer3);
	imshow("layer4", layer4);
	imshow("layer5", layer5);

	//���C���[���Ƃɕۑ�
	imwrite("layer0.png", layer0);
	imwrite("layer1.png", layer1);
	imwrite("layer2.png", layer2);
	imwrite("layer3.png", layer3);
	imwrite("layer4.png", layer4);
	imwrite("layer5.png", layer5);

	waitKey(0);
}

void sample8()
{
	using namespace cv;
	using namespace std;

	const char* input  = "sample_fg.jpg";
	const char* mask   = "mask.png";
	const char* output = "new.png";
	const char* mask2  = "layer0.png";

	cv::Mat image_src = cv::imread(input);
	cv::Mat image_mask = cv::imread(mask, 0);
	cv::Mat image_mask2 = cv::imread(mask2, 0);
	cv::Mat image_dest;

	cv::bitwise_not(image_mask, image_mask);
	cv::bitwise_not(image_mask2, image_mask2);

	// �f�މ摜���`�����l��(RGB)���Ƃɕ�������vector�Ɋi�[����
	std::vector<cv::Mat> mv;
	cv::split(image_src, mv);

	// ���ڗ̈�F�}�X�N�摜�̒��S�̑f�މ摜�̑傫���̗̈�
	cv::Rect rect((image_mask2.cols - image_src.cols) / 2,
		(image_mask2.rows - image_src.rows) / 2,
		image_src.cols,
		image_src.rows);
	// vector�̍Ō���Ƀ}�X�N�摜�̒��ڗ̈��ǉ�����
	mv.push_back(cv::Mat(image_mask2, rect));

	// vector���������ĉ��H��̉摜�Ƃ���
	cv::merge(mv, image_dest);

	// ���H��̉摜���o�͂���
	cv::imwrite(output, image_dest);


	imshow("layer0", image_mask2);
	waitKey(0);
}

int main(int argc, const char* argv[])
{
//	sample1();
//	sample2();
//	sample3();
//	sample4();
//	sample6();
//	sample7();
	sample8();

	return 0;
}