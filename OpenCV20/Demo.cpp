#include <all.h>
#include <fstream>
#include<opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

//#include<opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::ml;

void Demo::Demo01_Mat() {
	Mat x(Size(10, 10), CV_8UC1, Scalar(1));
	cout << x << endl;

	Mat b(Size(4, 4), CV_8UC3, Scalar(0, 0, 255));
	cout << b << endl;
	
	Mat e = Mat::eye(3,3,x.type());//单位矩阵 对角线
	cout << e << endl;

	Mat f(Size(1, 5), x.type(),Scalar(66));
	Mat g = Mat::diag(f);//对角矩阵  输入要是1*n的已有mat
	cout << g << endl;

	Mat a = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);//枚举创建
	cout << a << endl;

	Mat c = Mat::Mat(a, Range(1, 3), Range(1, 3));//截取 左闭右开  h w
	cout << c << endl;
}

void Demo::Demo02_Mat() {
	Mat a = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);//int 类型
	cout << a.at<int>(1, 1) << endl;

	Mat b(Size(4, 4), CV_8UC3, Scalar(0, 0, 255));
	//uchar 用b  根据通道数选择数字
	Vec3b vc = b.at<Vec3b>(0, 0); //此时是uchar类型
	cout << (int)vc[2] << endl;
}

void Demo::Demo03_Mat_Ope() {
	//矩阵乘积2*3 3*2 结果是矩阵 2*2
	//求内积dot是一个值  矩阵的数值的个数要一致1*6 2*3
	//对于位元素乘积mul 2*2 2*2 
	Mat a = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	Mat b = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 10);
	cout << a + b << endl;

	cout << a * b << endl;//线代的计算方法  矩阵类型要是浮点形 
	cout << a.dot(b) << endl;
	cout << a.mul(b) << endl;

	cout << min(a,b) << endl;
	cout << max(a, b) << endl;

}

void Demo::Demo04_Picture() {
	Mat src = imread("F:/test-picture/2.jpg",IMREAD_COLOR);//读取图像形式的标志
	Mat gray = imread("F:/test-picture/2.jpg", IMREAD_GRAYSCALE);

	namedWindow("Test1", WINDOW_AUTOSIZE);//WINDOW_AUTOSIZE 自适应大小
	namedWindow("Test2", WINDOW_NORMAL);
	imshow("Test1", src);
	imshow("Test2", gray);

	vector<int> compression_params;//官方示例
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	imwrite("F:/test-picture/3.png", gray, compression_params);
	waitKey(0);
	destroyAllWindows();
}

void Demo::Demo05_Watch() {
	Mat a = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);//注意下划线 

	Mat src = imread("F:/test-picture/2.jpg", IMREAD_COLOR);//读取图像形式的标志
	Mat gray = imread("F:/test-picture/2.jpg", IMREAD_GRAYSCALE);
}

void Demo::Demo06_Video() {
	VideoCapture video;
	video.open(0);
	if (!video.isOpened()) {
		cout << "打开视频失败" << endl;
		return;
	}
	cout << video.get(CAP_PROP_FPS) << endl;//fps
	cout << video.get(CAP_PROP_FRAME_WIDTH)<< endl;//宽度

	Mat frame;
	video >> frame;

	bool isColer = (frame.type() == CV_8UC3);
	VideoWriter writer;
	int codecc = VideoWriter::fourcc('M', 'J', 'P', 'G');//编码格式

	double fps = 25.0;
	string videoname = "F:/test-picture/test.avi";
	writer.open(videoname, codecc, fps, frame.size(), isColer);

	if (!writer.isOpened()) {
		cout << "打开视频失败" << endl;
		return;
	}

	while (true) {
		
		video >> frame;
		if (frame.empty()) {
			break;
		}

		imshow("video", frame);
		writer.write(frame);

		uchar c = waitKey(1000/ video.get(CAP_PROP_FPS));//两帧之间等待的时长为 1s/帧率    值大于 慢放 小于快放
		if (c == 'q') {
			break;
		}
	}
}

void Demo::Demo07_ColorChange() {
	//8U 0-255
	//32F 0-1
	//65F 0-1
	Mat img = imread("F:/test-picture/2.jpg", IMREAD_COLOR);
	Mat dis;
	img.convertTo(dis, CV_32F, 1/255.0, 0);//1/255.0 的确是用于缩放图像像素值的范围
	//想将图像的亮度提高一些，可以使用一个正数作为偏移量。如果你想降低亮度，可以使用一个负数作为偏移量

	Mat dis2,dis3;
	cvtColor(img, dis2, COLOR_BGR2HSV);
	cvtColor(dis, dis3, COLOR_BGR2HSV);

	imshow("test", dis);
	waitKey(0);
}

void Demo::Demo08_MulChannel() {
	namedWindow("Test1", WINDOW_AUTOSIZE);
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	imshow("Test1", img);

	
	Mat imgs[3];
	split(img, imgs);

	Mat img1 = imgs[0];
	Mat img2 = imgs[1];
	Mat img3 = imgs[2];

	Mat mer;
	merge(imgs, 2, mer);//只是合成两个通道
	
	vector<Mat> mats;
	Mat zero = Mat::zeros(img.size(), img.type());
	mats.push_back(img1);
	mats.push_back(zero);

	Mat dis;
	merge(mats, dis);

}

void Demo::Demo09_mimAndmax() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat temp=Mat::ones(img.size(),img.type());

	Mat mindis;
	Mat maxdis;
	min(img, temp,mindis);//两个图像的比较 大小和通道要相同
	max(img, temp,maxdis);


	cvtColor(img, img, COLOR_BGR2GRAY);
	double min;
	double max;

	Point minp;
	Point maxp;

	minMaxLoc(img, &min, &max, &minp, &maxp);//只能处理单通道图像  记得取地址
}

void Demo::Demo10_Logic() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat temp = Mat::ones(img.size(), img.type());

	Mat dis;
	bitwise_and(img, temp, dis,Mat());

}

void Demo::Demo11_binay() {//二值化
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat dis1;
	Mat dis2;
	threshold(img, dis1, 125,255, THRESH_BINARY);//对每个通道阈值化 再组合    255 在这里表示阈值化后的像素值的最大值
	threshold(img, dis2, 125, 255, THRESH_BINARY_INV);

	Mat adap;

	adaptiveThreshold(img,adap,255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,3,0);//只是支持灰度图像
	//adaptiveMethod：自适应阈值计算方法。可选值为 ADAPTIVE_THRESH_MEAN_C 和 ADAPTIVE_THRESH_GAUSSIAN_C，分别表示基于均值和高斯加权平均进行自适应阈值计算。
	//thresholdType：阈值类型，通常选择 THRESH_BINARY 或 THRESH_BINARY_INV，表示二值化或反二值化。
}

void Demo::Demo12_LUC() {//将原始的灰度值建立映射
	
	//查找表第一层
	uchar lutFirst[256];//建立灰度值从0-255的映射
	for (int i = 0; i < 256; i++) {
		if (i < 100) {
			lutFirst[i] = 0;
		}
		if (i > 100 && i <= 200) {
			lutFirst[i] = 100;
		}
		if (i > 200) {
			lutFirst[i] = 255;
		}
	}
	Mat lutOne(1, 256, CV_8UC1, lutFirst);//1 256

	//二
	uchar lutSecond[256];
	for (int i = 0; i < 256; i++) {
		if (i < 100) {
			lutFirst[i] = 0;
		}
		if (i > 100 && i <= 200) {
			lutFirst[i] = 50;
		}
		if (i > 200) {
			lutFirst[i] = 100;
		}
	}
	Mat lutTwo(1, 256, CV_8UC1, lutSecond);

	//三
	uchar lutThird[256];
	for (int i = 0; i < 256; i++) {
		if (i < 100) {
			lutFirst[i] = 100;
		}
		if (i > 100 && i <= 200) {
			lutFirst[i] = 200;
		}
		if (i > 200) {
			lutFirst[i] = 255;
		}
	}
	Mat lutThree(1, 256, CV_8UC1, lutThird);//通过传递 lutThird 数组给 Mat 构造函数，将数组的值复制到 lutTree 矩阵中

	//拥有三通道的LUT查找表
	vector<Mat> mergeMats;
	mergeMats.push_back(lutOne);
	mergeMats.push_back(lutTwo);
	mergeMats.push_back(lutThree);

	Mat LutTree;
	merge(mergeMats, LutTree);//记得要merge

	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat dis;
	Mat gray, out0, out1, out2;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	//LUT输入的图像只能是CV_8U..  范围只能是0-255的图像
	LUT(gray, lutOne, out0);//灰度为单通道 所以用单通道lut  
	LUT(img, lutOne, out1);//多对单
	LUT(img, LutTree, out2);//多对多

}

void Demo::Demo13_SizeChange() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat dis;
	//缩放
	resize(img, dis, Size(200, 200),0,0,INTER_AREA);//fx=2 fy=4 长变为2被 宽变为4备 与Size冲突时 选择Size来缩放 

	Mat x, y, xy;
	//翻转
	flip(img, x, 0);//>0 绕y轴 =0绕x轴 <0 绕两个轴
	flip(img, y, 1);
	flip(img, xy, -1);

	Mat mulx, muly;
	//拼接
	hconcat(img, img, mulx);//横向拼接 高要相同

	vconcat(img, img, muly);//纵向拼接 宽要相同
}

void Demo::Demo14_Rotat() {//平移 缩放 旋转 
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat rotation0,dis0;

	//1.获取仿射矩阵 2*3矩阵
	rotation0=getRotationMatrix2D(Point2f(img.rows/2.0, img.cols/2.0), -45.0, 0.5);//可以旋转和缩放 0.5 两个轴的比例因子

	warpAffine(img, dis0, rotation0,img.size());
	imshow("test", dis0);


	//2.知道原始和仿射后3点 可以计算仿射变换矩阵
	Point2f src_points[3];
	Point2f dis_points[3];
	src_points[0] = Point2f(0, 0);//原始点
	src_points[1] = Point2f(0, (float)(img.cols-1));
	src_points[2] = Point2f((float)(img.rows - 1), (float)(img.cols - 1));

	dis_points[0] = Point2f((float)(img.rows - 1)*0.11, (float)(img.cols - 1)*0.20);//变换后的点
	dis_points[1] = Point2f((float)(img.rows - 1)*0.15, (float)(img.cols - 1)*0.70);
	dis_points[2] = Point2f((float)(img.rows - 1)*0.81, (float)(img.cols - 1)*0.85);


	Mat rotation1, dis1;
	rotation1=getAffineTransform(src_points,dis_points);
	warpAffine(img, dis1, rotation1, img.size());

}

void Demo::Demo15_TouShi() {//透视变换（四点变换）  变换矩阵3*3
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);

	Point2f src_points[4];
	Point2f dis_points[4];

	Mat rotation0, dis0;
	rotation0=getPerspectiveTransform(src_points, dis_points);//获取透视变换矩阵

	warpPerspective(img,dis0,rotation0,img.size());//透视变换函数
}

void Demo::Demo16_Draw() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	line(img, Point(0, 0), Point(200, 200), Scalar(255, 0, 0), 3, LINE_AA, 0);//最后一个参数实现点的微偏移  因为点是整形 
	circle(img, Point(50, 50), 50, -1, 8, 0);
	ellipse(img, Point(100, 100), Size(50, 100),-45,0,180,Scalar(255),-1,8,0);//Size 设置中心距离长半轴和短半轴的大小 0 180为椭圆的角度
	rectangle(img, Point(100, 100), Point(200, 200), Scalar(100), -1, 8, 0);


	Mat temp = img.clone();
	Point pp[2][6];
	Point pp2[5];

	//const Point* pts[3] = { pp[0],pp[1],pp2 };
	//int npts[] = { 6,6,5 };

	////多边形绘制
	//fillPoly(temp,pts,npts,3,Scalar(255));//第二个参数为二维数组  每个子数组代表一个多边形   第三个参数为每个子数组的点的个数  第四个参数为绘制多边形的个数

	//生成文字
	putText(temp, "test", Point(50, 50),0,2,Scalar(255),5,8,false);//第三个参数为生成文字的左下角坐标  最后一个参数若为true  则图像原点在左下角（默认为左上角）
}

void Demo::Demo17_ROI() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	
	Mat dis, ROI1, ROI2;
	Rect rect(206, 206, 200, 200);  //起始坐标  长框
	ROI1 = img(rect);//截取

	ROI2=img(Range(0, 500), Range(0, 300));//截取 注意大写

	//dis=img(Rect(Point2i(200, 200), 200, 200));

	//深拷贝
	copyTo(img, dis, Mat());//cv类中
	img.copyTo(dis, Mat());//Mat类中

	//如果浅拷贝的图像应用到多个图像中 其中一个被修改 其它全部也会跟着被修改
}

void Demo::Demo18_PyrDown() {//下采样：缩小   将大的物体缩小 不断模糊 用于识别
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	resize(img, img, img.size() / 2, 0, 0);
	//imwrite("F:/test-picture/5.jpg", img);

	vector<Mat> Guass;
	int level = 3;
	Guass.push_back(img);
	for (int i = 0; i < level; i++) {
		Mat temp = Guass[i];//前一层
		Mat guass;
		pyrDown(temp, guass, temp.size() / 2);//最后一个参数为 填充方法
		Guass.push_back(guass);
	}

	for (int i = 0; i < level; i++) {
		String name = to_string(i);
		imshow(name, Guass[i]);
	}
	waitKey(0);
}

void Demo::Demo19_Pyr() {//残差金字塔   原图 - （原图下采样+上采样） 的图像   有bug。。。。。
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat dis;

	vector<Mat> Guass;
	int level = 3;
	Guass.push_back(img);
	for (int i = 0; i < level; i++) {
		Mat temp = Guass[i];//前一层
		Mat guass;
		pyrDown(temp, guass, temp.size() / 2);//最后一个参数为 填充方法
		Guass.push_back(guass);
	}

	vector<Mat> Lap;
	for (int i = level-1; i > 0; i--) {
		Mat lap, up;
		if (i == level - 1) {
			Mat down;
			pyrDown(Guass[i], down, Guass[i].size() / 2);//先缩小
			pyrUp(down, up, down.size() * 2);//后放大
			lap = Guass[i] - up;//原图-
		}
		else {//原本的金字塔有缩小的图像 ////？？？？？
			pyrUp(Guass[i], up, Guass[i].size() * 2);//放大
			lap = Guass[i - 1] - up;
		}
		Lap.push_back(lap);
	}
}

Mat img; //回调函数
void callBack(int value,void *) {//第一个参数是 createTrackbar中的value
	float a = value / 100;
	Mat img2 = img * a;
	imshow("img", img2);//还要显示在原来窗口
}

void Demo::Demo20_Bar() {
	img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	namedWindow("img", WINDOW_AUTOSIZE);
	imshow("img", img);

	int value = 50;
	createTrackbar("百分比", "img", &value, 655, callBack, 0);//初始值 最大值 
	waitKey(0);
}

Mat imgPoint;
Point prePoint;
void Mouse(int event, int x, int y, int flags, void* userdata) {//回调函数 envent 检测短时间操作按下键位  flags 检测长时间操作 拖拽 键盘按键
	if (event == EVENT_RBUTTONDOWN) {
		cout << "左键才能绘制" << endl;
	}
	if (event == EVENT_LBUTTONDOWN) {
		prePoint = Point(x, y);//记录起始坐标
		cout << "起始坐标" << prePoint << endl;
	}if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {//将flags与EVENT_FLAG_LBUTTON进行按位与运算，如果结果不为0，则表示左键被按下
		Point pt(x, y);
		line(img, prePoint, pt, Scalar(255), 2, 8, 0);
		prePoint = pt;
		imshow("图1", img);
	}
}

void Demo::Demo21_Mouse() {
	img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	img.copyTo(imgPoint);
	imshow("图1", img);
	imshow("图2", imgPoint);

	setMouseCallback("图1", Mouse, 0);
	waitKey(0);
}

//绘制直方图
void drawHist(Mat& hist, int type, string name) {//归一化并绘制直方图
	int hist_w = 512;
	int hist_h = 400;
	int width = 2;
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);//设置画板

	normalize(hist, hist, 1, 0, type, -1, Mat());//
	for (int i = 1; i <= hist.rows; i++) {
		rectangle(histImage, Point(width * (i - 1), hist_h - 1),
			Point(width * i - 1, hist_h - cvRound(hist_h * hist.at<float>(i - 1)) - 1),
			Scalar(255, 255, 255), -1);
	}
	imshow(name, histImage);
}

void Demo::Demo22_BarMap() {//
	img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat hist;//存放直方图统计结果
	int channels[1] = { 0 };//通道索引
	int bins[1] = { 256 };//灰度图的最大值
	const float inRanges[2] = { 0,255 };//灰度值变化范围
	const float* ranges[1] = { inRanges };//每个通道灰度值取值范围 这里就一个通道  记得加const！！！

	calcHist(&gray, 1, channels, Mat(), hist, 1, bins, ranges);// 图像 图像的数量 通道的索引（二维数组） mask 输出数组 每个维度直方图的数组尺寸（像素最大值）  
}

void Demo::Demo23_CalcHist() {//让像素分布更加广泛
	img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat equalImg;
	equalizeHist(gray, equalImg);//直方图均衡化

	Mat hist1,hist2;//存放直方图统计结果
	int channels[1] = { 0 };//通道索引
	int bins[1] = { 256 };//灰度图的最大值
	const float inRanges[2] = { 0,255 };//灰度值变化范围
	const float* ranges[1] = { inRanges };//每个通道灰度值取值范围 这里就一个通道  记得加const！！！

	calcHist(&gray, 1, channels, Mat(), hist1, 1, bins, ranges);
	calcHist(&equalImg, 1, channels, Mat(), hist2, 1, bins, ranges);

	drawHist(hist1, NORM_INF, "hist1");
	drawHist(hist2, NORM_INF, "hist2");

	imshow("原图",gray);
	imshow("均衡化", equalImg);

	waitKey(0);
}

void Demo::Demo24_Hist() {//直方图匹配 用原图像直方图的累计概率与目标直方图的累计概率比较  建立映射关系       有bug？？？
	Mat img1 = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat img2 = imread("F:/test-picture/6.jpg", IMREAD_COLOR);

	Mat hist1, hist2;
	int channels[1] = { 0 };//通道索引
	int bins[1] = { 256 };//灰度图的最大值
	const float inRanges[2] = { 0,255 };//灰度值变化范围
	const float* ranges[1] = { inRanges };//每个通道灰度值取值范围 这里就一个通道  记得加const！！！

	calcHist(&img1, 1, channels, Mat(), hist1, 1, bins, ranges);
	calcHist(&img2, 1, channels, Mat(), hist2, 1, bins, ranges);

	drawHist(hist1, NORM_INF, "hist1");
	drawHist(hist2, NORM_INF, "hist2");

	//计算两张图像直方图的累计概率
	float hist1_cdf[256] = { hist1.at<float>(0) };
	float hist2_cdf[256] = { hist2.at<float>(0) };

	for (int i = 1; i <= 255; i++) {
		hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);//累加
		hist2_cdf[i] = hist2_cdf[i - 1] + hist2.at<float>(i);
	}

	//for (int i = 0; i < 256; i++) {//test
	//	cout << hist1_cdf[i] << " ";
	//}
	//cout << endl;
	//for (int i = 0; i < 256; i++) {
	//	cout << hist2_cdf[i] << " ";
	//}
	//cout << endl;

	//构建累积概率误差矩阵
	float diff_cdf[256][256];
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			diff_cdf[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);//为了方便后序找出最小的
		}
	}

	//for (int i = 0; i < 256; i++) {//test
	//	for (int j = 0; j < 256; j++) {
	//		cout<<diff_cdf[i][j];
	//	}
	//	cout << endl;
	//}
	//cout << endl;

	//生成LUT映射表
	Mat lut(1, 256, CV_8UC1);

	for (int i = 0; i < 256; i++) {
		//查找灰度级为i的映射灰度
		//和i的累计概率差最小的规定为灰度
		float min = diff_cdf[i][0];
		int index = 0;//记录最小的下标(也是灰度)
		for (int j = 1; j < 256; j++) {
			if (min > diff_cdf[i][j]) {
				min = diff_cdf[i][j];
				index = j;
			}
		}
		lut.at<uchar>(i) = (uchar)index;
	}

	cout << lut << endl;

	Mat result, hist3;
	LUT(img1, lut, result);//bug finish
	imshow("原图", img1);
	imshow("匹配模板", img2);
	imshow("匹配结果", result);

	calcHist(&result, 1, channels, Mat(), hist3, 1, bins, ranges);
	drawHist(hist3, NORM_INF, "hist3");
	waitKey(0);
}

void Demo::Demo25_Match() {//模板匹配
	Mat img1 = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat img2 = imread("F:/test-picture/6.jpg", IMREAD_COLOR);
	Mat dis;
	matchTemplate(img1,img2,dis,TM_CCOEFF_NORMED,Mat());//输入为cv_8u  cv_32f 输出为cv32f

	double min;
	double max;
	Point minp;
	Point maxp;
	minMaxLoc(dis, &min, &max, &minp, &maxp);//在输出图像中寻找最大的值

	rectangle(img1, Point(maxp.x,maxp.y), Point(maxp.x + img2.cols, maxp.y + img2.rows),Scalar(255),5,8,0);
	waitKey(0);
}

void Demo::Demo26_filter() {
	Mat img1 = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat kernel = (Mat_<float>(3, 3) << 1, 2, 1,
		2, 0, 2,
		1, 2, 1);
	Mat kernel_norm = kernel / 12;//卷积模板归一化 防止卷积后大于255

	Mat result, result_norm;
	filter2D(img1, result, CV_32F, kernel, Point(-1, -1), 2, 4);//输出与输入有相同的尺寸和通道  输出的数据类型会发生改变  卷积核cv32fc1  anchor基准点（一般为卷积核中心位置） 偏移量  填充类型
	filter2D(img1, result_norm, CV_32F, kernel_norm, Point(-1, -1), 2, 4);
	imshow("result_norm", result_norm);//显示不出？？？
	waitKey(0);
}

void Demo::Demo27_Noise() {
	Mat img1 = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	//椒盐噪声需要手动添加 若是三通道 则每个通道都要添加
	cvflann::rand_double();
	cvflann::rand_int();

	//添加高斯噪声前要先创建一个与图像尺寸 数据类型和通道数量相同的mat类变量
	Mat noise = Mat::zeros(img1.size(), img1.type());
	//RNG::fill()
	RNG rng;//要先实例化
	rng.fill(noise, RNG::NORMAL, 10, 20);//均值和方差  创建噪声

	img1 = img1 + noise;//添加噪声
	imshow("111", img1);
	//imwrite("F:/test-picture/gaosi.jpg", img1);
	waitKey(0);
}

void Demo::Demo28_LinearFilter() {//线性滤波
	//均值滤波 
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat dis1,dis2,dis3;
	blur(img1, dis1, Size(3, 3), Point(-1, -1));//该函数强制进行归一化

	//方框滤波
	boxFilter(img1,dis3,-1,Size(3,3),Point(-1,-1),false);//和均值一致 可以选择是否归一化 false 不进行归一化

	//高斯滤波0
	GaussianBlur(img1, dis2, Size(15, 15), 0, 0);//如果size设置为0 则由标准差计算尺寸  x 和 y方向的滤波器标准偏差 如果为0 更具Size来计算
	waitKey(0);
}

void Demo::Demo29_MidFilter() {//中值滤波 有效去除椒盐噪声
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat dis;

	medianBlur(img1, dis, 7);
	waitKey(0);
}

void Demo::Demo30_SepFilter() {
	//线性滤波具有可分离性  先对x再对y  =  对x和y  能减少计算量
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat dis1,dis2,dis3,dis4,dis5;
	Mat a = (Mat_<float>(3, 1) << -1, 3, -1);
	Mat b = a.reshape(1,1);
	//cn：通道数，指定Mat对象的新通道数。如果为0，则表示保持原通道数不变。
	//rows：行数，指定Mat对象的新行数。
	Mat ab = a * b;

	filter2D(img1, dis1, -1, a, Point(-1, -1), 0);//-1表示输出图像的深度与输入图像相同
	filter2D(dis1, dis2, -1, b, Point(-1, -1), 0);

	filter2D(img1, dis3, -1, ab, Point(-1, -1), 0);//与上面两步骤结果一致  顺序不讲究

	sepFilter2D(img1, dis4 ,-1,a ,Mat(),Point(-1,-1),0);//第三个参数可以修改图像的数据类型 x方向滤波器 y方向滤波器
	sepFilter2D(img1, dis4, -1, Mat(), b, Point(-1, -1), 0);

	Mat gauss = getGaussianKernel(3,1);//尺寸和西格玛值 获取高斯滤波器
}

void Demo::Demo31_Edge1() {//边缘检测 一阶导的最大值（变化最大） f（x+1,y）-f(x-1,y)/2 
	//Sobel
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat disx,disy,disxy;


	//一般先行检测 后列检测
	Sobel(img1, disx, CV_16S,1,0,3);//CV_16S 因为相减可能会出现负数   dx dy表示对x和y轴求导的阶数 
	convertScaleAbs(disx, disx);//对边缘检测结果求取绝对值

	Sobel(img1, disy, CV_16S, 0, 1, 1);//ksize 输入1和3 是一致的
	convertScaleAbs(disy, disy);
	disxy = disx + disy;//倾斜的边缘可能会被加强
	imshow("x", disx);
	imshow("y", disy);
	imshow("Sobel", disxy);


	//Scharr
	Scharr(img1, disy, -1,1 , 0 , 1.0, 0.0);
	convertScaleAbs(disx, disx);
	Scharr(img1, disy, -1, 0, 1, 1.0, 0.0);
	convertScaleAbs(disy, disy);
	disxy = disx + disy;
	imshow("x", disx);
	imshow("y", disy);
	imshow("Sobel", disxy);
	waitKey(0);


	Mat sobelx, sobely, sobelX;
	Mat scharrx, scharry,scharrX;

	//获取一阶x方向的sobel算子  ???
	getDerivKernels(sobelx, sobely,1,0,3);//3*1
	cout << sobelx << endl;
	cout << sobely << endl;
	sobelx.reshape(CV_8U, 1);//1*3
	sobelX = sobely * sobelx;
	cout << sobelX << endl;

	//获取scharr算子
	getDerivKernels(scharrx, scharry, 1, 0, FILTER_SCHARR);
	scharrx.reshape(CV_8U, 1);//1*3
	scharrX = scharry * scharrx;
}

void Demo::Demo32_Edge2() {
	//Laplacian 无关方向提取边缘 容易受到噪声影响 
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	cvtColor(img1, img1, COLOR_BGR2GRAY);

	Mat dis1, dis2, dis3;
	Laplacian(img1, dis1, CV_16S , 3);
	convertScaleAbs(dis1, dis1);//具有负值像素的图像转换为无符号整数类型，即取绝对值并进行缩放
	
	//Canny 能够去除虚假边缘（光照等引起） 1.高斯平滑 2.梯度计算求方向和幅值 3.非极大值抑制 4.双阈值划分强 弱边缘 5.消除孤立弱边缘
	Canny(img1, dis2,100,200 ,3);// 两个阈值 sobel算子直径  输出是单通道图像 且只有0和255两个值cv_8u    

	//先滤波后Canny
	GaussianBlur(img1, dis3, Size(3, 3), 0, 0);
	Canny(dis3, dis3, 100, 200, 3);
	waitKey(0);
}

void Demo::Demo33_Connect() {//连通域分割
	//一般先二值化处理
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat rice,riceBW;
	cvtColor(img1, rice, COLOR_BGR2GRAY);//二值化之前先变成灰度图像
	threshold(rice, riceBW, 50, 255, THRESH_BINARY);

	RNG rng(1);
	Mat out;
	int num=connectedComponents(riceBW, out ,8,CV_16U);//输入为cv_8u  connectivity8为使用8联通区域  统计图像中连通域的个数  bug
	//输出中的不同的像素值表示的是不同的联通类  num就是分类的数值的个数

	vector<Vec3b> colors;//对每个一连通域都设置一个颜色
	for (int i = 0; i < num; i++) {
		Vec3b vec3 = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		colors.push_back(vec3);
	}

	Mat result = Mat::zeros(img1.size(), img1.type());//颜色通道为3 底色通道也为3
	int w = result.cols;
	int h = result.rows;
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			int label = out.at<uint16_t>(r, c);
			if (label == 0) {//黑色的不改变
				continue;
			}
			result.at<Vec3b>(r, c) = colors[label];//同一个数值 显示同一个颜色
		}
	}

	imshow("原图", img1);
	imshow("标记后图像", result);
	waitKey(0);


	//增加统计连通域信息
	Mat stats, centroids;
	num=connectedComponentsWithStats(riceBW,out,stats,centroids,8,CV_16U);//stats 不同连通域的统计信息矩阵 centroids 每个连通域的质心坐标  
	vector<Vec3b> new_colors;//对每个一连通域都设置一个颜色
	for (int i = 0; i < num; i++) {
		Vec3b vec3 = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		new_colors.push_back(vec3);
	}

	for (int i = 0; i < num; i++) {
		//中心位置
		int cen_x = centroids.at<double>(i, 0);//读取每个联通域的中心坐标
		int cen_y = centroids.at<double>(i, 1);

		//矩形边框
		int x = stats.at<int>(i, CC_STAT_LEFT);//最左边
		int y = stats.at<int>(i, CC_STAT_TOP);//最上
		int w = stats.at<int>(i, CC_STAT_WIDTH);//宽
		int h = stats.at<int>(i, CC_STAT_HEIGHT);//高

		//画中心点
		circle(img1, Point(cen_x, cen_y), 2, Scalar(0, 255, 0), 2);

		//外接矩阵
		Rect rect(x, y, w, h);
		rectangle(img1, rect, new_colors[i], 1);
		putText(img1, format("%d", i), Point(cen_x, cen_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
	}
	imshow("标记后", img1);
}

void Demo::Demo34_DisTran() {//计算图像中每个非0像素距离最近0像素(边界)的距离
	Mat a = (Mat_<uchar>(5, 5) << 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 0, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1);
	Mat dist_L1, dist_L2, dist_C, dist_L12;

	distanceTransform(a,dist_L1,1,3,CV_8U);//输入要为cv_8u的单通道图像 输出可以为cv_8u 或者cv_32f 1 街区距离 2欧式距离 3棋盘距离
	cout << dist_L1 << endl;

	distanceTransform(a, dist_L2, 2, 3, CV_8U);//5表示掩码尺寸为5x5 //
	cout << dist_L2 << endl;

	distanceTransform(a, dist_C, 3, 3, CV_8U);//5表示掩码尺寸为5x5
	cout << dist_C << endl;

	//waitKey(0);

	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat rice, riceBW, riceBW_INV;
	cvtColor(img1, rice, COLOR_BGR2GRAY);//二值化之前先变成灰度图像
	threshold(rice, riceBW, 50, 255, THRESH_BINARY);
	threshold(rice, riceBW_INV, 50, 255, THRESH_BINARY_INV);

	//距离变换
	Mat dist, dist_INV;
	distanceTransform(riceBW, dist, 1, 3, CV_32F);
	distanceTransform(riceBW_INV, dist_INV, 1, 3, CV_8U);

	waitKey(0);
}

void Demo::Demo35_Erode() {//形态学操作常用于对二值化图像操作 腐蚀常用于去除微小个体 分割粘连物体
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);

	//生成结构元素 //用于腐蚀/膨胀的卷积核
	Mat s1=getStructuringElement(0,Size(3,3),Point(-1,-1));//种类 大小 中心点
	Mat s2 = getStructuringElement(1, Size(3, 3));
	//0矩形 1十字型 2椭圆
	
	Mat dis1,dis2;
	erode(img1,dis1,s1);//itera 腐蚀次数
	erode(img1, dis2, s2,Point(-1,-1),10);

	waitKey(0);
}

void Demo::Demo36_Dilate() {//膨胀
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);

	//生成结构元素
	Mat s1 = getStructuringElement(0, Size(3, 3));//种类 大小 中心点
	Mat s2 = getStructuringElement(1, Size(3, 3));

	Mat dis1, dis2;
	dilate(img1, dis1, s1, Point(-1, -1), 5);
	dilate(img1, dis2, s2, Point(-1, -1), 10);

	waitKey(0);
}

void Demo::Demo37_Morpho() {
	//开运算 先腐蚀后膨胀 消除较小连通域（噪声）  保留较大连通域（再次膨胀）

	//闭运算 先膨胀后腐蚀  填充小空洞 连接两个临近连通域

	//形态学梯度 原图像膨胀-原图像腐蚀  突出图像中的边缘和物体的轮廓

	//顶帽 得到原图像中分散的一些斑点 原图-原图开运算  突出图像中比周围更亮的目标

	//黑帽 得到原图像中暗一点的区域  原图闭运算-原图  突出图像中比周围更暗的目标

	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	threshold(img1, img1, 80, 255, THRESH_BINARY);//形态选常用于二值化后图像
	Mat dis0,dis1, dis2, dis3, dis4, dis5, dis6,dis7;

	//获取结构元素
	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(img1,dis0,0,k);//0腐蚀 1膨胀 2开 3闭 4梯度 5顶帽 6黑帽 7击中 
	morphologyEx(img1, dis1, 1, k);
	morphologyEx(img1, dis2, 2, k);
	morphologyEx(img1, dis3, 3, k);
	morphologyEx(img1, dis4, 4, k);
	morphologyEx(img1, dis5, 5, k);
	morphologyEx(img1, dis6, 6, k);
	//morphologyEx(img1, dis7, 7, k);

	waitKey(0);
}

void Demo::Demo38_ximgproc() {//图像细化（骨架化）  一般是二值化图像 灰度也可以   拓展模块没安装好。。。烦死了。。。
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	threshold(img1, img1, 80, 255, THRESH_BINARY);//形态选常用于二值化后图像

	//ximgproc
}

void Demo::Demo39_Contours() {//轮廓
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat binary,gray;
	cvtColor(img1, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(13, 13), 4, 4);
	threshold(gray, binary, 50, 1, THRESH_BINARY);//形态选常用于二值化后图像  小于50的置0 大于的置255  
	//输入为单通道或者二值图像
	//cout << binary.type() << endl;

	vector<vector<Point>> contours;//轮廓 存储检测到的轮廓的向量容器 存储检测到的轮廓的向量容器
	vector<Vec4i> hierarchy;//存储轮廓的层次结构信息
	findContours(binary, contours, hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());//第二个参数为检测到的轮廓 存放着像素的坐标  bug???***********************  finish！！！

	for (int i = 0; i < hierarchy.size(); i++) {
		cout << hierarchy[i] << endl;
	}

	//轮廓绘制
	for (int t = 0; t < contours.size(); t++) {
		drawContours(img1, contours, t, Scalar(0, 0, 255), 2, 8);//t：指定要绘制的轮廓的索引 设置-1 绘制所有轮廓
		imshow("111", img1);
		waitKey(0);
	}
}

void Demo::Demo40_AreaAndLength() {
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat binary, gray;
	cvtColor(img1, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(13, 13), 4, 4);
	threshold(gray, binary, 50, 1, THRESH_BINARY);//形态选常用于二值化后图像  小于50的置0 大于的置255  

	vector<vector<Point>> contours;//轮廓 存储检测到的轮廓的向量容器 存储检测到的轮廓的向量容器
	vector<Vec4i> hierarchy;//存储轮廓的层次结构信息
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

	//输出轮廓面积
	for (int i = 0; i < contours.size(); i++) {
		double area1 = contourArea(contours[i]);//返回值为double
		cout << area1 << endl;
	}

	//输出轮廓长度
	for (int i = 0; i < contours.size(); i++) {
		double length1 = arcLength(contours[i],true);//返回值为double true表示计算的轮廓是闭合的
		cout << length1 << endl;
	}
}

void drawapp(Mat result, Mat img2) {
	for (int i = 0; i < result.rows; i++) {
		if (i == result.rows - 1) {
			Vec2i point1 = result.at<Vec2i>(i);
			Vec2i point2 = result.at<Vec2i>(0);

			line(img2, point1, point2, Scalar(0, 0, 255), 2, 8, 0);
			break;
		}

		Vec2i point1 = result.at<Vec2i>(i);
		Vec2i point2 = result.at<Vec2i>(i+1);
		line(img2, point1, point2, Scalar(0, 0, 255), 2, 8, 0);
	}
}

void Demo::Demo41_Fitting() {
	Mat img = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat img1, img2;//1最大外接 2最小外接
	img.copyTo(img1);
	img.copyTo(img2);

	Mat canny;
	Canny(img, canny, 80, 160, 3, false);//检测边缘

	Mat kernel = getStructuringElement(0, Size(3, 3), Point(-1, -1));
	dilate(canny, canny, kernel);//膨胀 将细小缝隙填补
	imshow("canny", canny);

	vector<vector<Point>> contours;//轮廓
	vector<Vec4i> hierarchy;//层次信息
	findContours(canny, contours, hierarchy, 0, 2, Point());

	//boundingRect();//返回值为矩形 得到最大外接矩形 输入为vecctor<Point>(一个轮廓) 或者Mat
	//minAreaRect();//返回值是旋转矩形RotateRect  
	//approxPolyDP();//多边形拟合
	for (int i = 0; i < contours.size(); i++) {
		Rect rect = boundingRect(contours[i]);//获取最大外接矩形
		rectangle(img1, rect, Scalar(0, 0, 255), 2, 8, 0);

		RotatedRect rrect = minAreaRect(contours[i]);//最小外接矩形  可以利用RotatedRect属性访问四个顶点(Point2f类型)和中心点
		Point2f points[4];
		rrect.points(points);//rrect.points() 函数的参数是一个 OutputArray 对象，它用于接收角点坐标
		Point2f cpt = rrect.center;//获取中心

		//绘制旋转矩阵与中心位置
		for (int i = 0; i < 4; i++) {
			if (i == 3) {
				line(img2, points[i], points[0], Scalar(0, 255, 0), 2, 8, 0);
				break;
			}
			line(img2, points[i], points[i + 1], Scalar(0, 255, 0), 2, 8, 0);
		}
		circle(img2, cpt, 4, Scalar(255, 0, 0), -1, 8, 0);
	}

	imshow("max", img1);
	imshow("min", img2);

	waitKey(0);

	cout << "多边形拟合" << endl;
	Mat approx = imread("F:/test-picture/mul.jpg");
	Mat canny2;
	Canny(approx, canny2, 80, 160, 3, false);
	Mat ker2 = getStructuringElement(0, Size(3, 3));
	dilate(canny2, canny2, ker2);

	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	findContours(canny2, contours2, hierarchy2, 0, 2, Point());

	for (int i = 0; i < contours2.size(); i++) {
		RotatedRect rrect = minAreaRect(contours2[i]);
		Point2f center = rrect.center;
		circle(approx, center, 2, Scalar(0, 0, 255), 2, 8, 0);

		Mat result;//n*2  可以用vector<Point>代替
		approxPolyDP(contours2[i], result, 4, true);//多边形拟合 result：表示输出的近似多边形的点序列  4：表示近似多边形的精度参数 越小越接近原始轮廓  true表示闭合
		drawapp(result, approx);//绘制轮廓
		cout << result.rows << endl;//点的个数

		if (result.rows == 3) {
			putText(approx, "triangle", center ,0, 1, Scalar(0, 255, 0), 1, 8);
		}
		else if (result.rows == 4) {
			putText(approx, "rectangle", center, 0, 1, Scalar(0, 255, 0), 1, 8);
		}
	}
	imshow("result", approx);
}

void Demo::Demo42_Hull() {//凸包检测
	Mat img = imread("F:/test-picture/handb.jpg");
	Mat gray, binary;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 50, 255, THRESH_BINARY);//阈值太大会导致轮廓过多

	//开运算消除噪声
	Mat k = getStructuringElement(0, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary,2, k, Point(-1, -1));
	imshow("binay", binary);

	//轮廓发现
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	findContours(binary, contours, 0, 2, Point());

	for (int n = 0; n < contours.size(); n++) {
		vector<Point> hull;//凸包
		//计算给定点集（轮廓）的凸包
		convexHull(contours[n], hull);
		for (int i = 0; i < hull.size(); i++) {
			circle(img, hull[i], 4, Scalar(255, 0, 0), 2, 8, 0);//在原图上绘制
			if (i == hull.size() - 1) {
				line(img, hull[i], hull[0], Scalar(0, 0, 255), 2, 8, 0);
				break;//直接break 否则下一行报错
			}
			line(img, hull[i], hull[i + 1], Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	imshow("hull", img);
	waitKey(0);
}

void drawLine(Mat& img, vector<Vec2f> lines, double rows, double cols, Scalar scalar, int n) {//原图像高 原图像宽 颜色 线宽
	//lines包含的是夹角以及直线距离坐标原点的距离
	Point pt1, pt2;//根据家教以及距离原点距离 先计算两个像素点
	for (int i=0; i < lines.size(); i++) {
		float rho = lines[i][0];//直线距离坐标原点的距离
		float theta = lines[i][1];//直线过坐标原点垂线于x轴夹角
		double co = cos(theta);
		double si = sin(theta);

		double x0 = rho * co;//焦点
		double y0 = rho * si;
		double length = max(rows, cols);

		pt1.x = cvRound(x0 + length*(-si)); //cvRound对浮点数进行四舍五入取整
		pt1.y = cvRound(y0 + length * (co));

		pt2.x = cvRound(x0 - length * (-si));//???
		pt2.y = cvRound(y0 - length * (co));

		line(img, pt1, pt2, scalar, n);
	}
}

void Demo::Demo43_HoughLine() {//直线检测 原空间一个点（可以多条线穿过） -> 参数空间一条线（线中的每个点 就是原空间过点的一条直线）
							//									多个点 -> 多条线 （若两条直线相交 交点对应的直线 对应是原空间中过两点的直线
							//									（过三点的直线  -> 每个点对应直线相交于一点）
							//									原空间直线经过的点=参数空间交点被直线经过的次数（设置阈值 寻找直线）

	//1.参数空间离散化 2.映射 3.统计每个方格出现的次数 选取大于某一阈值的方格作为表示直线的方格 4.将参数空间表示直线的方格参数作为图像中直线的参数
	Mat img = imread("F:/test-picture/box.jpg", IMREAD_GRAYSCALE);
	Mat edge;
	Canny(img, edge, 80, 180, 3, false);
	threshold(edge, edge, 170, 255, THRESH_BINARY);//先边缘检测 再直线检测

	vector<Vec2f> lines1, lines2;
	vector<Vec4i> linesP3, linesP4;
	HoughLines(edge, lines1, 1, CV_PI / 180, 50,  0, 0);//1离散化的单位  50阈值只有达到才视为是直线 
	HoughLines(edge, lines2, 1, CV_PI / 180, 150, 0, 0);

	HoughLinesP(edge, linesP3, 1, CV_PI / 180, 150, 30, 10);//150 阈值（影响数量） 30最下线段长度，小于舍去  10线段间允许的最大间隔（影响长短）
	HoughLinesP(edge, linesP4, 1, CV_PI / 180, 150, 30, 30);//得到的结果linesP4就是坐标  

	Mat img1, img2,img3,img4;
	img.copyTo(img1);
	img.copyTo(img2);
	img.copyTo(img3);
	img.copyTo(img4);
	drawLine(img1,lines1,edge.rows,edge.cols,Scalar(255),2);
	drawLine(img2, lines2, edge.rows, edge.cols, Scalar(255), 2);

	for (int i = 0; i < linesP3.size(); i++) {
		line(img3, Point(linesP3[i][0], linesP3[i][1]), Point(linesP3[i][2], linesP3[i][3]), Scalar(255), 3);
	}

	for (int i = 0; i < linesP4.size(); i++) {
		line(img4, Point(linesP4[i][0], linesP4[i][1]), Point(linesP4[i][2], linesP4[i][3]), Scalar(255), 3);
	}

	waitKey(0);
}

void Demo::Demo44_Enclose() {//对离散点进行拟合
	//Vec4f lines;
	//vector<Point2f> points;

	//double parm = 0;
	//double reps = 0.01;//坐标原点与直线之间的距离精度
	//double aeps = 0.01;//角度精度

	////直线拟合 2d或者3d点集  2d点集描述参数为vec4f 3d为vec6f 
	//fitLine(points, lines, DIST_L1, 0, reps, aeps);//parm:如果 distType 为 CV_DIST_L2，则该参数为 0；如果 distType 为 CV_DIST_L1，则该参数为 0.01
	////lines (vx, vy, x0, y0)  方向向量(vx, vy)表示直线的方向(x和y方向分量)， (x0, y0) 则表示直线上的一个点。
	//double k = lines[1] / lines[0];//斜率

	//waitKey(0);

	Mat img(500, 500, CV_8UC3, Scalar::all(0));//Scalar::all(0) 返回一个所有分量都为 0 的 Scalar 对象
	RNG& rng = theRNG();//// 获取默认的随机数生成器对象  相较于RNG rng 这种方法有关联

	while (true)
	{
		int i, count = rng.uniform(1, 101);
		vector<Point> points;
		//生成随机点
		for (int i = 0; i < count; i++) {
			Point pt;
			pt.x = rng.uniform(img.cols / 4, img.cols * 3 / 4);
			pt.y = rng.uniform(img.rows / 4, img.rows * 3 / 4);
			points.push_back(pt);
		}

		//拟合三角形
		vector<Point2f> triangle;
		double area = minEnclosingTriangle(points, triangle);//返回值表示找到的最小三角形的面积

		//拟合圆
		Point2f center;
		float radius = 0;
		minEnclosingCircle(points, center, radius);

		//创建两图像用于输出
		img = Scalar(0);//清0
		Mat img2;
		img.copyTo(img2);

		//绘制点
		for (int i = 0; i < count; i++) {
			circle(img, points[i], 3, Scalar(255, 255, 255), FILLED, LINE_AA);
			circle(img2, points[i], 3, Scalar(255, 255, 255), FILLED, LINE_AA);
		}

		//绘制三角形
		for (int i = 0; i < 3; i++) {
			if (i == 2) {
				line(img, triangle[i], triangle[0], Scalar(255, 255, 0), 1, 16);
				break;
			}
			line(img, triangle[i], triangle[i+1], Scalar(255, 255, 0), 1, 16);
		}

		//绘制圆形
		circle(img2, center, radius, Scalar(255, 255, 0), 1, LINE_AA);

		imshow("triangle", img);
		imshow("circle", img2);

		char key = (char)waitKey();
		if (key == 27 || key == 'q') {
			break;
		}
	}
}

void Demo::Demo45_Code() {//二维码识别
	Mat img = imread("F:/test-picture/code.jpg", IMREAD_COLOR);
	Mat gray, qrcode;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	QRCodeDetector qrcodee;//先实例化
	vector<Point> points;
	string info;
	bool isQRcode;
	isQRcode = qrcodee.detect(gray, points);//获取二维码位置信息

	if (isQRcode) {
		info = qrcodee.decode(gray, points, qrcode);//解码二维码 qrcode提取二维码
		cout << points << endl;
	}
	for(int i=0;i<points.size();i++){
		if (i == points.size() - 1) {
			line(img, points[i], points[0], Scalar(0, 255, 255), 2, 8);
			break;
		}
		line(img, points[i], points[i+1], Scalar(0, 255, 255), 2, 8);
	}

	putText(img, info, Point(20, 30), 0, 1.0, Scalar(0, 0, 255), 2, 8);

	//定位并解析函数
	string info2;
	vector<Point> points2;
	info2 = qrcodee.detectAndDecode(gray, points2);
	cout << points2 << endl;

	putText(img, info2.c_str(), Point(20, 55), 0, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("img", img);
	imshow("qrcode", qrcode);
}

void Demo::Demo46_Integral() {//积分图像 加速图像像素处理过程（类似于提前打表）
	Mat img = Mat::ones(16, 16, CV_32FC1);

	RNG rng(22);
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			float d = rng.uniform(-0.5, 0.5);
			img.at<float>(y, x) = img.at<float>(y, x) + d;
		}
	}

	Mat sum,sqsum,tilted;
	integral(img, sum,sqsum,tilted);//标准求和 平方求和 三角形求和
	Mat sum8U = Mat_<uchar>(sum);
	Mat sqsum8U = Mat_<uchar>(sqsum);//
	Mat tilted8U = Mat_<uchar>(tilted);//bug ??? finish

	waitKey(0);
}

void Demo::Demo47_floodFill() {//漫水填充 1.选种子点 2.判断4或者8领域内像素值和种子值差值 小于阈值的添加进区域 3.新加的像素点作为新的种子点
	Mat img = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	RNG rng(22);

	int connectivity = 4;//联通领域方式
	int maskVal = 255;//掩码图像数值  掩码图像用于标记已访问的像素。255 是一个常见的选项，表示将掩码图像中填充过的区域设置为最大值
	int flags = connectivity | (maskVal << 8) | FLOODFILL_FIXED_RANGE;//填充操作方式标志

	//设置与选中像素点的差值  上下都浮动20
	Scalar loDiff = Scalar(20, 20, 20);
	Scalar upDiff = Scalar(20, 20, 20);

	Mat mask = Mat::zeros(img.rows + 2, img.cols+2, CV_8UC1);//用于记录哪些位置被填充过

	while (true)
	{
		int py = rng.uniform(0, img.rows - 1);
		int px = rng.uniform(0, img.cols - 1);
		Point point = Point(px, py);

		Scalar newVal = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		int area = floodFill(img, mask, point, newVal,nullptr, loDiff, upDiff, flags); //newVal填充值 Rect返回填充区域的边界矩阵 

		cout << point.x << " " << point.y << " " << area << endl;
		imshow("img", img);

		int c = waitKey(0);
	}
}

void Demo::Demo48_Watershed() {//分水岭 1.排序 对局部最小注水 2.淹没 寻找汇集区域 就是分割线
	Mat img, imgGray, imgMask, img_;//imgMask 分割线
	Mat maskWaterShed;//Watershed函数参数
	img = imread("F:/test-picture/wline.png");//带有标记图像
	img_ = imread("F:/test-picture/5.jpg");
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	////二值化并开运算
	threshold(imgGray, imgMask, 254, 255, THRESH_BINARY);
	Mat k = getStructuringElement(0, Size(3, 3));
	morphologyEx(imgMask, imgMask, 1, k);

	imshow("标记图像", img);
	imshow("原图像", img_);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgMask, contours, hierarchy, 0, 2);

	//markers 的作用就是我们预先把一些区域标注好，这些标注了的区域称之为种子点。watershed 算法会把这些标记的区域慢慢膨胀填充整个图像
	maskWaterShed = Mat::zeros(img.size(), CV_32S);//找0 1 2
	
	for (int i = 0; i < contours.size(); i++) {
		drawContours(maskWaterShed, contours, i, Scalar::all(i+1), -1, 8, hierarchy, INT_MAX);//把每一个轮廓内区域全部填充为1 2 3...  为什么？？？
	}

	RNG rng;
	//分水岭算法 对原图像处理
	watershed(img_, maskWaterShed);//mask要求要是CV_32S
	vector<Vec3b> colors;
	for (int i = 0; i < contours.size(); i++) {
		colors.push_back(Vec3b((uchar)rng.uniform(0, 255), (uchar)rng.uniform(0, 255), (uchar)rng.uniform(0, 255)));
	}

	Mat resultImg = Mat(img.size(), CV_8UC3);//显示图像
	for (int i = 0; i < imgMask.rows; i++) {
		for (int j = 0; j < imgMask.cols; j++) {
			int index = maskWaterShed.at<int>(i, j);
			if (index == -1) {//边界
				resultImg.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else if (index <= 0 || index > contours.size()) {//没有标记的区域置0
				resultImg.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else {
				resultImg.at<Vec3b>(i, j) = colors[index - 1];
			}
		}
	}

	imshow("result", resultImg);
	//resultImg = resultImg * 0.8 + img_ * 0.2;//bug 类型不匹配。。。
	addWeighted(resultImg, 0.8, img_, 0.2, 0, resultImg);//
	imshow("result", resultImg);

	for (int n = 1; n < contours.size(); n++) {
		Mat resImage = Mat(img.size(), CV_8UC3);
		for (int i = 0; i < imgMask.rows; i++) {
			for (int j = 0; j < imgMask.cols; j++) {
				int index = maskWaterShed.at<int>(i, j);
				if (index == n) {
					resImage.at<Vec3b>(i, j) = img_.at<Vec3b>(i, j);
				}
				else {
					resImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);//bug Vec3d。。。
				}
			}
		}
		//
		imshow(to_string(n), resImage);
	}

	waitKey(0);
}

void Demo::Demo49_Harris() {//Harris角点检测 只能输入灰度图像
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	//计算Harris系数
	Mat harris;
	int blockSize = 2;//角点检测时要考虑的领域大小。这个值越大，检测到的角点往往也越多。
	int kSize = 3;//sobel算子大小
	cornerHarris(gray, harris, blockSize, kSize, 0.04);//k的取值一般为0.02-0.04

	Mat harrisn;
	normalize(harris, harrisn, 0, 255, NORM_MINMAX);//将图像归一化到0-255
	convertScaleAbs(harrisn, harrisn);//将数据类型变为CV_8U
	//将图像进行缩放、求绝对值并转换为 8 位无符号整数格式

	//harris每个像素的数值代表了该位置处的角点响应强度。角点响应强度的数值越大，表示该位置越可能是角点。

	//寻找harris角点
	vector<KeyPoint> keyPoints;
	for (int row = 0; row < harrisn.rows; row++) {
		for (int col = 0; col < harrisn.cols; col++) {
			int R = harrisn.at<uchar>(row, col);
			if (R > 208) {
				KeyPoint keyPoint;
				keyPoint.pt.x = row;//注意.pt
				keyPoint.pt.y = col;
				keyPoints.push_back(keyPoint);
			}
		}
	}

	drawKeypoints(img, keyPoints, img);
	imshow("系数矩阵", harrisn);
	imshow("角点",img);

	waitKey(0);
}

void Demo::Demo50_KeyPoint() {//Shi-Tomas角点
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	int maxCorners = 100;//检测角点的数目
	double quality_level = 0.01;//质量等级
	double minDistance = 0.04;//两个角点之间的最小欧式距离
	vector<Point2f> corners;
	goodFeaturesToTrack(gray, corners, maxCorners, quality_level, minDistance, Mat(), 3, false);//false 表示不是计算海瑞思角点
	//返回值就是角点的坐标

	vector<KeyPoint> keyPoints;
	for (int i = 0; i < corners.size(); i++) {
		KeyPoint keyPoint;
		keyPoint.pt = corners[i];
		keyPoints.push_back(keyPoint);
	}

	drawKeypoints(img, keyPoints, img);
	imshow("角点", img);
	waitKey(0);
}

void Demo::Demo51_SubPix() {//角点位置亚像素优化

	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	int maxCorners = 100;//检测角点的数目
	double quality_level = 0.01;//质量等级
	double minDistance = 0.04;//两个角点之间的最小欧式距离
	vector<Point2f> corners;
	goodFeaturesToTrack(gray, corners, maxCorners, quality_level, minDistance, Mat(), 3, false);
	//使用 TermCriteria::EPS 和 TermCriteria::COUNT 两个常量进行按位或操作来指定终止条件的类型。
	//使用的终止条件是 TermCriteria::EPS + TermCriteria::COUNT，表示终止条件同时满足迭代次数和目标函数值的变化。

	Size winSize = Size(5, 5);
	Size zeroSize = Size(-1, -1);//死区参数 一般设置为-1 -1
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);
	cornerSubPix(gray, corners, winSize, zeroSize, criteria);
	
	vector<KeyPoint> keyPoints;
	for (int i = 0; i < corners.size(); i++) {
		KeyPoint kp;
		kp.pt = corners[i];
		keyPoints.push_back(kp);
	}

	drawKeypoints(img,keyPoints,img);
	imshow("111", img);
}

void Demo::Demo52_ORB() {//特征点计算 ORB特征点（计算资源消耗少）
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Ptr<ORB> orb = ORB::create(500);//其余使用默认值

	//计算关键点（还没描述子）
	vector<KeyPoint> KeyPoints;
	orb->detect(img, KeyPoints);

	//计算ORB描述子
	Mat description;
	orb->compute(img, KeyPoints, description);

	//绘制特征点
	Mat imgAngel;
	drawKeypoints(img, KeyPoints, imgAngel, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("111", imgAngel);
	waitKey(0);
}

void orb_features(Mat& gray, vector<KeyPoint>& keypoints, Mat& des) {
	Ptr<ORB> orb = ORB::create(1000, 1.2f);
	orb->detect(gray, keypoints);
	orb->compute(gray, keypoints, des);
}

void Demo::Demo53_FeatureMatch() {//特征点匹配
	Mat img1, img2;
	img1 = imread("F:/test-picture/p1.jpg");
	img2 = imread("F:/test-picture/p2.jpg");

	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;

	//计算特征点
	orb_features(img1, kp1, des1);
	orb_features(img2, kp2, des2);

	vector<DMatch> matches;//存放匹配结果的变量
	BFMatcher matcher(NORM_HAMMING);//定义特征点匹配类 使用汉明距离作为特征描述符之间的差异度量（适用于二值特征描述符，如ORB、BRIEF等）  暴力匹配
	matcher.match(des1,des2,matches);//特征点匹配

	cout << "matches" << matches.size() << endl;

	//找到汉明距离最小和最大值
	double min_dist = 10000, max_dist = 0;
	for (int i = 0; i < matches.size(); i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
		if (dist > max_dist) {
			max_dist = dist;
		}
	}

	cout << "min" << min_dist << " " << "max" << max_dist << endl;

	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance <= max(2 * min_dist, 20.0)) {//只保留汉明距离较小的点
			good_matches.push_back(matches[i]);
		}
	}

	cout << good_matches.size() << endl;

	//绘制匹配结果
	Mat outimg, outimg1;
	drawMatches(img1, kp1, img2, kp2, matches, outimg);
	drawMatches(img1, kp1, img2, kp2, good_matches, outimg1);

	imshow("未筛选结果", outimg);
	imshow("筛选结果", outimg1);
	waitKey(0);
}

void match_min(vector<DMatch>& matches, vector<DMatch>& good_min) {
	double min_dist = 10000, max_dist = 0;
	for (int i = 0; i < matches.size(); i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
		if (dist > max_dist) {
			max_dist = dist;
		}
	}

	cout << "min" << min_dist << " " << "max" << max_dist << endl;

	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance <= max(2 * min_dist, 20.0)) {//只保留汉明距离较小的点
			good_min.push_back(matches[i]);
		}
	}
}

//RANSAC算法实现
void ransac(vector<DMatch>& matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint, vector<DMatch>& match_ransac) {
	//定义保存匹配点对坐标
	vector<Point2f> srcPoints(matches.size()), dstPoints(matches.size());

	//保存从关键点中提取到的匹配点对的坐标
	for (int i = 0; i < matches.size(); i++) {
		srcPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
		//srcPoints是一个存储了特征点坐标的数组，queryKeyPoint是查询图像中的特征点集合，matches是匹配结果的向量。
		//matches[i].queryIdx表示匹配对中的第i个匹配对的查询图像中的特征点的索引
		dstPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;//一句话--在特征点集合查找匹配对的点
	}

	vector<int> inlinersMask(srcPoints.size());
	findHomography(srcPoints, dstPoints, RANSAC, 5, inlinersMask);//计算两幅图像之间的单应性矩阵  bug
	//它的输入参数包括srcPoints（查询图像中的特征点坐标）、dstPoints（参考图像中的特征点坐标）、method（计算单应性矩阵的方法
	//RANSAC是一种常见的方法）、ransacReprojThreshold（RANSAC算法中的阈值）和mask（输出的掩码，表示哪些点被认为是内点）等。
	//RANSAC方法可有效地排除掉误匹配的特征点，从而得到更加准确的单应性矩阵。
	//inlinersMask中的每个元素都被设置为0或1，其中0表示对应的特征点是外点（即不符合单应性变换模型），而1表示对应的特征点是内点（即符合单应性变换模型）。

	//转成DMatch形式
	for (int i = 0; i < inlinersMask.size(); i++) {
		if (inlinersMask[i]) {//只有为1的才push
			match_ransac.push_back(matches[i]);
		}
	}
}

void Demo::Demo54_Ransac() {//ransac
	Mat img1, img2;
	img1 = imread("F:/test-picture/p1.jpg");
	img2 = imread("F:/test-picture/p2.jpg");

	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;

	//计算特征点
	orb_features(img1, kp1, des1);
	orb_features(img2, kp2, des2);

	//特征点匹配
	vector<DMatch> matches, good_min, good_ransac;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(des1, des2, matches);
	cout << matches.size() << endl;

	//初次筛选
	match_min(matches, good_min);
	cout << good_min.size() << endl;//0??? fished

	//ransac算法筛选
	ransac(good_min, kp1, kp2, good_ransac);

	Mat outimg, outimg1, outimg2;
	drawMatches(img1, kp1, img2, kp2, matches, outimg);
	drawMatches(img1, kp1, img2, kp2, good_min,outimg1);
	drawMatches(img1, kp1, img2, kp2, good_ransac, outimg2);

	//imshow("未筛选", outimg);
	//imshow("最小汉明筛选", outimg1);
	//imshow("ransac筛选", outimg2);
	waitKey(0);
}

void Demo::Demo55_Camera() {//相机模型
	Mat cameraMatrix;//内参矩阵 f dx dy
	Mat disCoeffs;//畸变矩阵 k1 k2 k3 p1 p2

	Mat rvec;//旋转  注意是Mat类型
	Mat tvec;//平移  从世界转到相机坐标系

	vector<Point3f> PointSets;//空间的点 3d
	vector<Point2f> imagePoints;//映射到像素空间的点
	projectPoints(PointSets, rvec, tvec, cameraMatrix, disCoeffs, imagePoints);
}

void Demo::Demo56_CameraFind() {//相机标定
	vector<Mat> imgs;//标定板图片
	string imageName;
	ifstream fin("F:/test-picture/cailbdata.txt");
	while (getline(fin, imageName)) {//添加标定板图片
		Mat img = imread(imageName);
		imgs.push_back(img);
	}

	Size board_size = Size(9, 6);//标定板内角点数目（行 列）
	vector<vector<Point2f>> imgsPoints;//每一张图片的角点坐标（多张）
	for (int i = 0; i < imgs.size(); i++) {
		Mat img1 = imgs[i];
		Mat gray1;
		cvtColor(img1, gray1, COLOR_BGR2GRAY);
		vector<Point2f> img1_points;
		findChessboardCorners(gray1, board_size, img1_points);//检测图像中棋盘格模式的角点
		find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));//对初始的角点坐标进行亚像素级别的优化
		bool pattern = true;
		drawChessboardCorners(img1, board_size, img1_points, pattern);//绘制检测到的棋盘格角点
		imshow("img1", img1);
		waitKey(0);
		imgsPoints.push_back(img1_points);
	}

	Size squareSize = Size(10, 10);//假设棋盘格子每个方格真实尺寸
	vector<vector<Point3f>> objectPoints;//真实世界坐标
	for (int i = 0; i < imgsPoints.size(); i++) {
		vector<Point3f> tempPointSet;
		for (int j = 0; j < board_size.height; j++) {//为每个角点设置坐标
			for (int k = 0; k < board_size.width; k++) {
				Point3f realPoint;
				realPoint.x = j * squareSize.width;
				realPoint.y = k * squareSize.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(tempPointSet);
	}
	
	//图像尺寸
	Size imageSize;
	imageSize.width = imgs[0].cols;
	imageSize.height = imgs[0].rows;

	Mat cameraMatrix = Mat(Size(3, 3), CV_32FC1);//内参矩阵
	Mat distCoeffs = Mat(1, 5, CV_32FC1);//畸变矩阵 k1 k12 k3 p1 p2
	vector<Mat> rvecs;//每张图的旋转向量  每幅图像对应唯一的外参矩阵
	vector<Mat> tvecs;//每张图的平移量
	calibrateCamera(objectPoints, imgsPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);//求取内参矩阵和畸变矩阵
}

//对多张图片矫正函数
void undist(vector<Mat> imgs, Mat cameraMatrix, Mat distCoeffs, vector<Mat>& undistImgs) {
	for (int i = 0; i < imgs.size(); i++) {
		Mat undistImg;
		undistort(imgs[i], undistImg, cameraMatrix, distCoeffs);
		undistImgs.push_back(undistImg);
	}
}

void Demo::Demo57_Camera() {//图像矫正
	//undist();


}

void Demo::Demo58_Pnp() {//单目位姿估计 （确定相机在世界坐标系中的位置和朝向）
	Mat img1,gray1;
	Size board_size = Size(9, 6);//标定板内角点数目（行 列）
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	vector<Point2f> imgPoints;//角点
	findChessboardCorners(gray1, board_size, imgPoints);//检测图像中棋盘格模式的角点
	find4QuadCornerSubpix(gray1, imgPoints, Size(5, 5));//对初始的角点坐标进行亚像素级别的优化

	vector<Point3f> PointSets;//角点对应的世界坐标系位置(上一个位置相机坐标系的点)
	Mat cameraMatrix = Mat(Size(3, 3), CV_32FC1);//内参矩阵
	Mat distCoeffs = Mat(1, 5, CV_32FC1);//畸变矩阵 k1 k12 k3 p1 p2
	Mat rvec, tvec;
	solvePnP(PointSets, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);//求解旋转向量和平移向量 

	Mat R;
	Rodrigues(rvec, R);//旋转向量变成旋转矩阵

	//Pnp+Ransac计算
	Mat Ranrvec, Rantvec;
	solvePnPRansac(PointSets, imgPoints, cameraMatrix, distCoeffs, Ranrvec, Rantvec);

}

void Demo::Demo59_Absdiff() {//差值法检测移动物体  当前时刻图像-前一时刻图像/背景图像=入侵物体
	VideoCapture capture(0);
	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);//总帧率

	Mat preFrame, preGray;
	capture.read(preFrame);//读取第一帧作为背景
	cvtColor(preFrame, preGray, COLOR_BGR2GRAY);
	GaussianBlur(preGray, preGray, Size(0, 0), 15);

	Mat binary;
	Mat frame, gray;
	Mat k = getStructuringElement(0, Size(7, 7), Point(-1, -1));

	while (true)
	{
		if (!capture.read(frame)) {
			break;
		}
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, gray, Size(0, 0), 15);
		absdiff(gray, preGray, binary);//当前帧-背景帧率
		threshold(binary, binary, 10, 255,THRESH_BINARY|THRESH_OTSU);
		morphologyEx(binary, binary, 2, k);//开运算

		imshow("input", frame);
		imshow("result", binary);

		//gray.copyTo(preFrame);//使用前一帧当背景

		char c = waitKey(5);
		if (c == 27) {
			break;
		}
	}
}

void Demo::Demo60_FlowFarneback() {//稠密光流法跟踪物体（计算量大）
	//前提假设 1.同一个物体对应的像素亮度不变
	//2.两帧图像必须较小的运动
	//3.区域运动具有一致性（某个区域中每个像素的变化规律一致）
	VideoCapture catpure(0);
	Mat preFrame, preGray;

	catpure.read(preFrame);
	cvtColor(preFrame, preGray, COLOR_BGR2GRAY);

	while (true)
	{
		Mat nextFrame, nextGray;
		if (!catpure.read(nextFrame)) {
			break;
		}
		imshow("原始", nextFrame);
		cvtColor(nextFrame, nextGray, COLOR_BGR2GRAY);

		Mat_<Point2f> flows;//是一个矩阵，每个元素都是一个Point2f类型的点。
		calcOpticalFlowFarneback(preGray, nextGray, flows, 0.5, 3, 15, 3, 5, 1.2, 0);//计算两个方向的运动速度
		Mat xV = Mat::zeros(preFrame.size(), CV_32FC1);//x方向速度
		Mat yV = Mat::zeros(preFrame.size(), CV_32FC1);//y方向速度

		for (int r = 0; r < flows.rows; r++) {
			for (int c = 0; c < flows.cols; c++) {
				const Point2f& flow_xy = flows.at<Point2f>(r, c);//返回Point2f的点
				//常量引用表示不能修改被绑定的对象，因此 flow_xy 是一个只读的引用，用于访问 flows 中的数据
				xV.at<float> (r, c)= flow_xy.x;
				yV.at<float>(r, c) = flow_xy.y;
			}
		}

		//计算向量的角度和幅值(信号或图像的振幅或强度,表示了信号或图像在某个特定点的大小或强度)
		Mat magnitude, angle;//magnitude存储计算得到的向量的模（极坐标中的距离）,angle存储计算得到的向量的角度（极坐标中的角度）
		cartToPolar(xV, yV, magnitude, angle);//将笛卡尔坐标系下的向量转换为极坐标系  将输入的 x 和 y 坐标分量组合成一个向量，并计算该向量的模和角度

		//弧度转成角度制
		angle = angle * 180.0 / CV_PI / 2.0;

		//把幅值归一化到0-255区间便于显示
		normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);

		convertScaleAbs(magnitude, magnitude);//取绝对值
		convertScaleAbs(angle, angle);

		Mat HSV = Mat::zeros(preFrame.size(), preFrame.type());//创建的是HSV？？？
		vector<Mat> result;
		split(HSV, result);
		result[0] = angle;//???  颜色0-180
		result[1] = Scalar(255);//饱和度
		result[2] = magnitude;//亮度
		merge(result, HSV);

		Mat rgbImg;
		cvtColor(HSV, rgbImg, COLOR_HSV2BGR);

		imshow("运动检测结果", rgbImg);
		int ch = waitKey(5);
		if (ch == 27) {
			break;
		}
	}
}

void draw_lines(Mat &image,vector<Point2f> pt1,vector<Point2f> pt2) {
	RNG rng(22);
	for (int i = 0; i < pt1.size(); i++) {
		line(image, pt1[i], pt2[i], Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2, 8, 0);
	}
}
void Demo::Demo61_FlowPyrLK() {//稀疏光流法跟踪
	VideoCapture catpure(0);
	Mat preFrame, prevImg;

	catpure.read(preFrame);
	cvtColor(preFrame, prevImg, COLOR_BGR2GRAY);

	vector<Point2f> Points;//检测第一帧角点
	int maxCorners = 100;//检测角点的数目
	double quality_level = 0.01;//质量等级
	double minDistance = 10;//两个角点之间的最小欧式距离
	int blockSize = 3;
	goodFeaturesToTrack(prevImg, Points, maxCorners, quality_level, minDistance, Mat(), blockSize , false);//false 表示不是计算海瑞思角点

	//稀疏光流检测相关参数设置
	vector<Point2f> prevPts;//前一帧图像角点坐标
	vector<Point2f> nextPts;//当前帧图像角点坐标
	vector<uchar> status;//检测到的状态
	vector<float> err;
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);//迭代的最大步数  目标函数的精度
	double derivlambda = 0.5;
	int flags = 0;

	//初始化状态的角点
	//vector<Point2f> initPoints;
	//initPoints.insert(initPoints.end(), Points.begin(), Points.end());

	//前一帧图像中的角点坐标
	prevPts.insert(prevPts.end(), Points.begin(), Points.end());

	while (true)
	{
		Mat nextframe, nextImg;
		if (!catpure.read(nextframe)) {
			break;
		}
		imshow("原图", nextframe);

		cvtColor(nextframe, nextImg, COLOR_BGR2GRAY);
		//光流跟踪
		calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, Size(31, 31), 3, criteria, derivlambda, flags);

		//cout << "status:";//test
		//for (int i = 0; i < status.size(); i++) {
		//	cout << status[i] << " ";
		//}
		//cout << endl;

		//cout << prevPts.size() << endl;//test
		//cout << nextPts.size() << endl;
		//cout << initPoints.size() << endl;

		//检测角点是否移动 不移动就删除
		size_t i, k;
		for (i = k = 0; i < nextPts.size(); i++) {//bug...
			//距离与状态测量
			double dist = abs(prevPts[i].x - nextPts[i].x) + abs(prevPts[i].y - nextPts[i].y);
			if (status[i] && dist > 2) {//检测到 且距离有移动 才“入列”
				prevPts[k] = prevPts[i];
				//initPoints[k] = initPoints[i];
				nextPts[k] = nextPts[i];//bug vector越界？？？ finish
				k++;
				circle(nextframe, nextPts[i], 3, Scalar(0, 255, 0), -1, 8);
			}
		}

		//更新移动角点数目
		prevPts.resize(k);
		nextPts.resize(k);
		//initPoints.resize(k);

		//绘制跟踪轨迹
		//draw_lines(nextframe, initPoints, nextPts);
		draw_lines(nextframe, prevPts, nextPts);
		imshow("result", nextframe);

		char c = waitKey(50);
		if (c == 27) {
			break;
		}

		swap(nextPts, prevPts);//主要是为了更新prev  next无所谓
		nextImg.copyTo(prevImg);

		//if (initPoints.size() < 30) {
		if (prevPts.size() < 30) {
			goodFeaturesToTrack(prevImg, Points, maxCorners, quality_level, minDistance, Mat(), blockSize, false);
			//initPoints.insert(initPoints.end(), Points.begin(), Points.end());//
			prevPts.insert(prevPts.end(), Points.begin(), Points.end());
		}
	}
}

void Demo::Demo62_Knn() {
	Mat img;//假设图像有5000个数字样本 每个数500个 每个数据大小位20*20

	//转灰度

	Mat images = Mat::zeros(5000,400,CV_8UC1);
	Mat labels = Mat::zeros(5000, 1, CV_8UC1);

	int index = 0;//记录处理的个数

	Rect numberImg;//把每个20*20 转成1*400  用作访问images媒介
	numberImg.x=0;//获取该矩形区域左上角点的 x 坐标
	numberImg.height = 1;
	numberImg.width = 400;

	//images(numberImg);

	for (int r = 0; r < 50; r++) {//10个数 每个数5行
		int label = r / 5;
		int datay = r * 20;//每个数字左上角坐标
		for (int c = 0; c < 100; c++) {//一行100个
			int datax = c * 20;
			Mat number = Mat::zeros(Size(20, 20), CV_8UC1);
			for (int x = 0; x < 20; x++) {//遍历每个数字的20*20方格
				for (int y = 0; y < 20; y++) {
					number.at<uchar>(x, y) = img.at<uchar>(datay+x, datax+y);
				}
			}

			//将二维图像转成行数据
			Mat row = number.reshape(1, 1);
			numberImg.y = index;

			//添加到总数据
			row.copyTo(images(numberImg));
			//images(numberImg) 则是使用括号运算符来访问 images 矩阵中的子区域，通过 numberImg 来指定感兴趣的矩形区域。
			labels.at<uchar>(index, 0) = label;//列向量
			index++;
		}
	}

	images.convertTo(images, CV_32FC1);//类型转换函数
	labels.convertTo(labels, CV_32SC1);

	Ptr<ml::TrainData> tdata = ml::TrainData::create(images, ml::ROW_SAMPLE, labels);//ROW_SAMPLE每一行代表一个样本，也就是每一行包含了一个完整的样本的所有特征 COL_SAMPLE每列
	//指向 ml::TrainData 对象的智能指针 tdata

	//创建K近邻类
	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(5);//考虑最近的5个样本
	knn->setIsClassifier(true);//true用作分类 false用作回归

	knn->train(tdata);
	//保存训练结果
	String path;//.yml
	knn->save(path);
	waitKey(0);

	/*******************************************************************************/

	//加载模型并用于训练
	Mat datas;
	Mat labels;
	Ptr<KNearest> knn = Algorithm::load<KNearest>("knn_model.yml");

	//分类
	Mat result;
	knn->findNearest(datas, 5, result);

	Mat test1;//用于测试图片
	Mat test2;

	resize(test1, test1, Size(20, 20));
	resize(test2, test2, Size(20, 20));
	Mat one = test1.reshape(1, 1);
	Mat tow = test2.reshape(1, 1);

	Mat testdata = Mat::zeros(2, 400, CV_8UC1);
	Rect rect;//作为中间变量
	rect.x = 0;
	rect.y = 0;
	rect.height = 1;
	rect.width = 400;
	
	one.copyTo(testdata(rect));
	rect.y = 1;
	tow.copyTo(testdata(rect));

	//进行估计识别
	Mat result2;//2*1 
	knn->findNearest(testdata, 5, result2);
	waitKey(0);
}

//void Demo::Demo63_SVM() {
//	Mat samples, lables;
//	//FileStorage用于读取和写入 XML、YAML 和 JSON 等格式的文件，用于保存和加载机器学习模型、特征向量等数据
//	FileStorage fread("point.yml", FileStorage::READ);//FileStorage::READ 标志，指示进行读取操作
//	fread["data"] >> samples;
//	fread["lables"] >> lables;
//	fread.release();
//
//	vector<Vec3b> colors;
//
//	Mat img;//创建空白图像显示坐标点
//	Mat img2;
//
//	Ptr<SVM> model = SVM::create();
//
//	model->setKernel(SVM::INTER);//内核模型
//	model->setType(SVM::C_SVC);//SVM类型
//	model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001));
//	//...
//	model->train(TrainData);
//	
//	model->predict();
//}

void Demo::Demo64_Camera() {

}
                                                        
void Demo::Demo65_Net() {//加载神经网络模型
	string model;
	string config;

	Net net = dnn::readNet(model, config);//加载模型
	vector<String> layerName = net.getLayerNames();
	for (int i = 0; i < layerName.size(); i++) {
		int ID = net.getLayerId(layerName[i]);
		//读取每层网络信息
		Ptr<Layer> layer = net.getLayer(ID);
		cout << layer->type.c_str() << endl;
	}
}

void Demo::Demo66_Net() {

}