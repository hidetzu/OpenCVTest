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

void sendThread() {
	SOCKET s, s1;         //�\�P�b�g
	int result;          //�߂�l
						 //�ڑ���������N���C�A���g�[���̏��
	struct sockaddr_in source;
	char buffer[1024];  //��M�f�[�^�̃o�b�t�@�̈�
	char ans[] = "���M����";
	char ret;

	memset(&buffer, '\0', sizeof(buffer));

	//���M���̒[������o�^����
	memset(&source, 0, sizeof(source));
	source.sin_family = AF_INET;

	//�|�[�g�ԍ��̓N���C�A���g�v���O�����Ƌ���
	source.sin_port = htons(7000);
	source.sin_addr.s_addr = htonl(INADDR_ANY);

	//�\�P�b�g�̐���
	s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (s < 0) {
		printf("%d\n", GetLastError());
		printf("�\�P�b�g�����G���[\n");
	}

	//�\�P�b�g�̃o�C���h
	result = bind(s, (struct sockaddr *)&source, sizeof(source));
	if (result < 0) {
		printf("%d\n", GetLastError());
		printf("�o�C���h�G���[\n");
	}

	//�ڑ��̋���
	result = listen(s, 1);
	if (result < 0) {
		printf("�ڑ����G���[\n");
	}

	printf("�ڑ��J�n\n");
	//�N���C�A���g����ʐM������܂őҋ@
	s1 = accept(s, NULL, NULL);
	if (s1 < 0) {
		printf("�ҋ@�G���[\n");
	}

	cv::Mat frameReference = cv::imread("lena.png", cv::IMREAD_COLOR);
	std::vector<uchar> buf;

	std::vector<int> params = std::vector<int>(2);
	params[0] = CV_IMWRITE_PNG_COMPRESSION;
	params[1] = 9;

	cv::imencode(".png", frameReference, buf, params);

	auto imgSize = buf.size();
	send(s1, (char*)&imgSize, sizeof(int64_t), 0);
	send(s1, (char*)(&buf[0]), imgSize, 0);

	printf("�ڑ��I��\n");
	closesocket(s1);
}

void encode_decode()
{
	WSADATA data;
	WSAStartup(MAKEWORD(2, 0), &data);

	std::thread th(sendThread);
	SOCKET s;    //�\�P�b�g
				 //�ڑ�����T�[�o�̏��
	struct sockaddr_in dest;
	char destination[] = "127.0.0.1";
	char buffer[1024];

	//�ڑ���i�T�[�o�j�̃A�h���X����ݒ�
	memset(&dest, 0, sizeof(dest));

	//�|�[�g�ԍ��̓T�[�o�v���O�����Ƌ���
	dest.sin_port = htons(7000);
	dest.sin_family = AF_INET;
	dest.sin_addr.s_addr = inet_addr(destination);

	//�\�P�b�g�̐���
	s = socket(AF_INET, SOCK_STREAM, 0);

	//�T�[�o�ւ̐ڑ�
	if (connect(s, (struct sockaddr *) &dest, sizeof(dest))) {
		printf("%s�ɐڑ��ł��܂���ł���\n", destination);
		return;
	}

	printf("%s�ɐڑ����܂���\n", destination);

	int64_t bufSize;
	recv(s, (char*)&bufSize, sizeof(int64_t), 0);

	uchar*  buffer2 = new uchar[bufSize];

	//�T�[�o����f�[�^����M
	int bytes = 0;
	for (int i = 0; i < bufSize; i += bytes) {
		if ((bytes = recv(s, (char*)(buffer2 + i), bufSize - i, 0)) == -1) {
		}
	}

	std::vector<uchar> buf;

	for (int i = 0; i < bufSize; i++) {
		buf.push_back(buffer2[i]);
	}

	closesocket(s);

	/// �f�R�[�h(from jpeg200)
	// �o�b�t�@�Cimread�Ɠ����t���O
	cv::Mat dst_img = cv::imdecode(cv::Mat(buf), 1);

	cv::namedWindow("both flip image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::imshow("both flip image", dst_img);
	cv::waitKey(0);

	th.join();

	WSACleanup();
}

void sample2()
{
	//�摜�̓ǂݍ���
	cv::Mat img_src1 = cv::imread("sora.jpg", cv::IMREAD_COLOR);
	cv::Mat gori(cv::Size(img_src1.cols, img_src1.rows), CV_8UC3);
	int bytes = img_src1.total() * img_src1.elemSize();

	uchar* data = new uchar[bytes];
	memcpy(data, img_src1.data, bytes);

#if false
	for (int j = 0; j<height; j++)
	{
		int step = j*width;
		for (int i = 0; i<width; i++)
		{
			int elm = i*img_src1.elemSize();
			for (int c = 0; c<channels; c++)
			{
				data[step + elm + c] = img_src1.data[step + elm + c];
			}
		}
	}
#endif
	gori.data = data;

	cv::namedWindow("both flip image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::imshow("both flip image", gori);
	cv::imshow("�摜", img_src1);

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
		cv::Vec3b* imgPtr  = image.ptr<cv::Vec3b>(y);
		uchar* keyImgPtr = keyImage.ptr<uchar>(y);
		for (int x = 0; x < image.cols; ++x)
		{
			auto imgBgr = imgPtr[x];
			d = (imgBgr[0] + imgBgr[2]) / 2 - imgBgr[1];

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
				auto rr = (unsigned char)((rr1*kk + rr2*(255 - kk)) / 255);
				auto gg = (unsigned char)((gg1*kk + gg2*(255 - kk)) / 255);
				auto bb = (unsigned char)((bb1*kk + bb2*(255 - kk)) / 255);
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

void sample6()
{
	//�摜�̓ǂݍ���
	cv::Mat fgImg = cv::imread("sample_fg.jpg", cv::IMREAD_COLOR);
	cv::Mat bgImg = cv::imread("sample_fg_bg.jpg", cv::IMREAD_COLOR);
	cv::Mat img_back(cv::Size(fgImg.cols, fgImg.rows), CV_8UC3, cv::Scalar::all(255));
	cv::Mat result = fgImg.clone();

	cv::Mat img_src1_gray;
	cv::Mat img_src2_gray;
	cv::cvtColor(fgImg, img_src1_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(bgImg, img_src2_gray, cv::COLOR_BGR2GRAY);

	cv::Mat matDiff;
	cv::absdiff(fgImg, bgImg, matDiff);
	cv::cvtColor(matDiff, matDiff, cv::COLOR_BGR2GRAY);
	cv::imwrite("matDiff.png", matDiff);

	cv::Mat binaryImg = matDiff.clone();
	soft_mask(matDiff, binaryImg, 1, 10);
//	cv::threshold(matDiff, binaryImg, 40, 255, CV_THRESH_BINARY_INV);
	cv::imwrite("binary.png", binaryImg);

	cv::Mat keyImg = cv::Mat(fgImg.rows, fgImg.cols, CV_8UC1, cv::Scalar(255));
	s_synth(fgImg, img_back, binaryImg, &result);
	cv::imwrite("result.jpg", result);
}

int main(int argc, const char* argv[])
{
//	sample1();
//	encode_decode();
//	sample2();
//	sample3();
//	sample4();
	sample6();

	return 0;
}