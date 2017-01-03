// OpenCVTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "MaskMethod.h"
#include "ThresholdMethod.h"
#include "Composite.h"

int main(int argc, const char* argv[])
{
	//画像の読み込み
	cv::Mat img_src1 = cv::imread("sora.jpg", cv::IMREAD_COLOR);
	cv::Mat img_src2 = cv::imread("dambo3.jpg", cv::IMREAD_COLOR);
	if (img_src1.empty()) 
		return -1;

	if (img_src2.empty())
		return -1;

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

	return 0;
}