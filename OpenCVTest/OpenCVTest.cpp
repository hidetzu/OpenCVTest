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
	cv::Mat img_src1 = cv::imread("dummy_1.png", cv::IMREAD_COLOR);
	//cv::Mat img_src2 = cv::imread("sample_bg.jpg", cv::IMREAD_COLOR);
	cv::Mat img_src2(cv::Size(img_src1.cols, img_src1.rows), CV_8UC3, cv::Scalar::all(255));
	if (img_src1.empty())
		return;

	if (img_src2.empty())
		return;

	cv::Mat hsv;
	cv::cvtColor(img_src1, hsv, cv::COLOR_BGR2HSV);

	//OpenCV HSV H : 0 - 180, S : 0 - 255, V : 0 - 255
	//	lower_color = np.array([100, 110, 30]) # �F��Ԃ̉���
	//	upper_color = np.array([120, 255, 255]) # �F��Ԃ̏��
	cv::Mat mask;
	cv::inRange(hsv,
		cv::Scalar(40,  0,  30, 0),
		cv::Scalar(100, 255, 255, 0),
		mask);

	// �m�C�Y����
#if true
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
	cv::imshow("Show MASK COMPOSITION Image", result);

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

int main(int argc, const char* argv[])
{
//	sample1();
//	encode_decode();
//	sample2();
	sample3();

	return 0;
}