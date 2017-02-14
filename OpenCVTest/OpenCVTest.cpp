// OpenCVTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
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
	//画像の読み込み
	cv::Mat img_src1 = cv::imread("sora.jpg", cv::IMREAD_COLOR);
	cv::Mat img_src2 = cv::imread("dambo3.jpg", cv::IMREAD_COLOR);
	if (img_src1.empty())
		return;

	if (img_src2.empty())
		return;

	// TODO: 二値化の方法は固定閾値法を利用する。
	auto maskFactory = new ThresholdMethod(245);
	Composite* composite = new Composite(img_src2, img_src1, maskFactory);
	cv::Mat result = composite->Execute();

	// 結果の表示
	cv::imshow("Show MASK COMPOSITION Image", result);
	delete composite;
	composite = nullptr;

	cv::waitKey(0);
	cv::destroyAllWindows();
}

void sendThread() {
	SOCKET s, s1;         //ソケット
	int result;          //戻り値
						 //接続を許可するクライアント端末の情報
	struct sockaddr_in source;
	char buffer[1024];  //受信データのバッファ領域
	char ans[] = "送信成功";
	char ret;

	memset(&buffer, '\0', sizeof(buffer));

	//送信元の端末情報を登録する
	memset(&source, 0, sizeof(source));
	source.sin_family = AF_INET;

	//ポート番号はクライアントプログラムと共通
	source.sin_port = htons(7000);
	source.sin_addr.s_addr = htonl(INADDR_ANY);

	//ソケットの生成
	s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (s < 0) {
		printf("%d\n", GetLastError());
		printf("ソケット生成エラー\n");
	}

	//ソケットのバインド
	result = bind(s, (struct sockaddr *)&source, sizeof(source));
	if (result < 0) {
		printf("%d\n", GetLastError());
		printf("バインドエラー\n");
	}

	//接続の許可
	result = listen(s, 1);
	if (result < 0) {
		printf("接続許可エラー\n");
	}

	printf("接続開始\n");
	//クライアントから通信があるまで待機
	s1 = accept(s, NULL, NULL);
	if (s1 < 0) {
		printf("待機エラー\n");
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

	printf("接続終了\n");
	closesocket(s1);
}

void encode_decode()
{
	WSADATA data;
	WSAStartup(MAKEWORD(2, 0), &data);

	std::thread th(sendThread);
	SOCKET s;    //ソケット
				 //接続するサーバの情報
	struct sockaddr_in dest;
	char destination[] = "127.0.0.1";
	char buffer[1024];

	//接続先（サーバ）のアドレス情報を設定
	memset(&dest, 0, sizeof(dest));

	//ポート番号はサーバプログラムと共通
	dest.sin_port = htons(7000);
	dest.sin_family = AF_INET;
	dest.sin_addr.s_addr = inet_addr(destination);

	//ソケットの生成
	s = socket(AF_INET, SOCK_STREAM, 0);

	//サーバへの接続
	if (connect(s, (struct sockaddr *) &dest, sizeof(dest))) {
		printf("%sに接続できませんでした\n", destination);
		return;
	}

	printf("%sに接続しました\n", destination);

	int64_t bufSize;
	recv(s, (char*)&bufSize, sizeof(int64_t), 0);

	uchar*  buffer2 = new uchar[bufSize];

	//サーバからデータを受信
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

	/// デコード(from jpeg200)
	// バッファ，imreadと同じフラグ
	cv::Mat dst_img = cv::imdecode(cv::Mat(buf), 1);

	cv::namedWindow("both flip image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::imshow("both flip image", dst_img);
	cv::waitKey(0);

	th.join();

	WSACleanup();
}

void sample2()
{
	//画像の読み込み
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
	cv::imshow("画像", img_src1);

	cv::waitKey(0);
	cv::destroyAllWindows();
}

void sample3()
{
	//画像の読み込み
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
	//	lower_color = np.array([100, 110, 30]) # 色空間の下限
	//	upper_color = np.array([120, 255, 255]) # 色空間の上限
	cv::Mat mask;
	cv::inRange(hsv,
		cv::Scalar(40,  0,  30, 0),
		cv::Scalar(100, 255, 255, 0),
		mask);

	// ノイズ除去
#if true
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  cv::Mat(), cv::Point(-1, -1), 1);
#endif

	cv::imwrite("mask.png", mask);

	// 人物画を抽出
	cv::Mat portraitPhotographTmp;
	{
		cv::Mat notImg;
		cv::bitwise_not(mask, notImg);
		img_src1.copyTo(portraitPhotographTmp, notImg);
		cv::imwrite("portraitPhotographTmp.jpg", portraitPhotographTmp);
	}

	// 背景画を抽出 
	cv::Mat backgroundPictureTmp;
	{
		img_src2.copyTo(backgroundPictureTmp, mask);
		cv::imwrite("backgroundPictureTmp.jpg", backgroundPictureTmp);
	}

	// 人物画と背景画を合成する。
	cv::Mat result;
	cv::bitwise_or(backgroundPictureTmp, portraitPhotographTmp, result);
	cv::imshow("Show MASK COMPOSITION Image", result);

#if false
	cv::Mat notImg;
	cv::Mat result;
	cv::bitwise_not(mask, notImg); // マスクを反転
	cv::imwrite("notmask.png", notImg);
	cv::bitwise_and(img_src1, img_src1, result, notImg);

	// 結果の表示
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