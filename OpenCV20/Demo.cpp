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
	
	Mat e = Mat::eye(3,3,x.type());//��λ���� �Խ���
	cout << e << endl;

	Mat f(Size(1, 5), x.type(),Scalar(66));
	Mat g = Mat::diag(f);//�ԽǾ���  ����Ҫ��1*n������mat
	cout << g << endl;

	Mat a = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);//ö�ٴ���
	cout << a << endl;

	Mat c = Mat::Mat(a, Range(1, 3), Range(1, 3));//��ȡ ����ҿ�  h w
	cout << c << endl;
}

void Demo::Demo02_Mat() {
	Mat a = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);//int ����
	cout << a.at<int>(1, 1) << endl;

	Mat b(Size(4, 4), CV_8UC3, Scalar(0, 0, 255));
	//uchar ��b  ����ͨ����ѡ������
	Vec3b vc = b.at<Vec3b>(0, 0); //��ʱ��uchar����
	cout << (int)vc[2] << endl;
}

void Demo::Demo03_Mat_Ope() {
	//����˻�2*3 3*2 ����Ǿ��� 2*2
	//���ڻ�dot��һ��ֵ  �������ֵ�ĸ���Ҫһ��1*6 2*3
	//����λԪ�س˻�mul 2*2 2*2 
	Mat a = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	Mat b = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 10);
	cout << a + b << endl;

	cout << a * b << endl;//�ߴ��ļ��㷽��  ��������Ҫ�Ǹ����� 
	cout << a.dot(b) << endl;
	cout << a.mul(b) << endl;

	cout << min(a,b) << endl;
	cout << max(a, b) << endl;

}

void Demo::Demo04_Picture() {
	Mat src = imread("F:/test-picture/2.jpg",IMREAD_COLOR);//��ȡͼ����ʽ�ı�־
	Mat gray = imread("F:/test-picture/2.jpg", IMREAD_GRAYSCALE);

	namedWindow("Test1", WINDOW_AUTOSIZE);//WINDOW_AUTOSIZE ����Ӧ��С
	namedWindow("Test2", WINDOW_NORMAL);
	imshow("Test1", src);
	imshow("Test2", gray);

	vector<int> compression_params;//�ٷ�ʾ��
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	imwrite("F:/test-picture/3.png", gray, compression_params);
	waitKey(0);
	destroyAllWindows();
}

void Demo::Demo05_Watch() {
	Mat a = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);//ע���»��� 

	Mat src = imread("F:/test-picture/2.jpg", IMREAD_COLOR);//��ȡͼ����ʽ�ı�־
	Mat gray = imread("F:/test-picture/2.jpg", IMREAD_GRAYSCALE);
}

void Demo::Demo06_Video() {
	VideoCapture video;
	video.open(0);
	if (!video.isOpened()) {
		cout << "����Ƶʧ��" << endl;
		return;
	}
	cout << video.get(CAP_PROP_FPS) << endl;//fps
	cout << video.get(CAP_PROP_FRAME_WIDTH)<< endl;//���

	Mat frame;
	video >> frame;

	bool isColer = (frame.type() == CV_8UC3);
	VideoWriter writer;
	int codecc = VideoWriter::fourcc('M', 'J', 'P', 'G');//�����ʽ

	double fps = 25.0;
	string videoname = "F:/test-picture/test.avi";
	writer.open(videoname, codecc, fps, frame.size(), isColer);

	if (!writer.isOpened()) {
		cout << "����Ƶʧ��" << endl;
		return;
	}

	while (true) {
		
		video >> frame;
		if (frame.empty()) {
			break;
		}

		imshow("video", frame);
		writer.write(frame);

		uchar c = waitKey(1000/ video.get(CAP_PROP_FPS));//��֮֡��ȴ���ʱ��Ϊ 1s/֡��    ֵ���� ���� С�ڿ��
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
	img.convertTo(dis, CV_32F, 1/255.0, 0);//1/255.0 ��ȷ����������ͼ������ֵ�ķ�Χ
	//�뽫ͼ����������һЩ������ʹ��һ��������Ϊƫ������������뽵�����ȣ�����ʹ��һ��������Ϊƫ����

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
	merge(imgs, 2, mer);//ֻ�Ǻϳ�����ͨ��
	
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
	min(img, temp,mindis);//����ͼ��ıȽ� ��С��ͨ��Ҫ��ͬ
	max(img, temp,maxdis);


	cvtColor(img, img, COLOR_BGR2GRAY);
	double min;
	double max;

	Point minp;
	Point maxp;

	minMaxLoc(img, &min, &max, &minp, &maxp);//ֻ�ܴ���ͨ��ͼ��  �ǵ�ȡ��ַ
}

void Demo::Demo10_Logic() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat temp = Mat::ones(img.size(), img.type());

	Mat dis;
	bitwise_and(img, temp, dis,Mat());

}

void Demo::Demo11_binay() {//��ֵ��
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat dis1;
	Mat dis2;
	threshold(img, dis1, 125,255, THRESH_BINARY);//��ÿ��ͨ����ֵ�� �����    255 �������ʾ��ֵ���������ֵ�����ֵ
	threshold(img, dis2, 125, 255, THRESH_BINARY_INV);

	Mat adap;

	adaptiveThreshold(img,adap,255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,3,0);//ֻ��֧�ֻҶ�ͼ��
	//adaptiveMethod������Ӧ��ֵ���㷽������ѡֵΪ ADAPTIVE_THRESH_MEAN_C �� ADAPTIVE_THRESH_GAUSSIAN_C���ֱ��ʾ���ھ�ֵ�͸�˹��Ȩƽ����������Ӧ��ֵ���㡣
	//thresholdType����ֵ���ͣ�ͨ��ѡ�� THRESH_BINARY �� THRESH_BINARY_INV����ʾ��ֵ���򷴶�ֵ����
}

void Demo::Demo12_LUC() {//��ԭʼ�ĻҶ�ֵ����ӳ��
	
	//���ұ��һ��
	uchar lutFirst[256];//�����Ҷ�ֵ��0-255��ӳ��
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

	//��
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

	//��
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
	Mat lutThree(1, 256, CV_8UC1, lutThird);//ͨ������ lutThird ����� Mat ���캯�����������ֵ���Ƶ� lutTree ������

	//ӵ����ͨ����LUT���ұ�
	vector<Mat> mergeMats;
	mergeMats.push_back(lutOne);
	mergeMats.push_back(lutTwo);
	mergeMats.push_back(lutThree);

	Mat LutTree;
	merge(mergeMats, LutTree);//�ǵ�Ҫmerge

	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat dis;
	Mat gray, out0, out1, out2;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	//LUT�����ͼ��ֻ����CV_8U..  ��Χֻ����0-255��ͼ��
	LUT(gray, lutOne, out0);//�Ҷ�Ϊ��ͨ�� �����õ�ͨ��lut  
	LUT(img, lutOne, out1);//��Ե�
	LUT(img, LutTree, out2);//��Զ�

}

void Demo::Demo13_SizeChange() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat dis;
	//����
	resize(img, dis, Size(200, 200),0,0,INTER_AREA);//fx=2 fy=4 ����Ϊ2�� ���Ϊ4�� ��Size��ͻʱ ѡ��Size������ 

	Mat x, y, xy;
	//��ת
	flip(img, x, 0);//>0 ��y�� =0��x�� <0 ��������
	flip(img, y, 1);
	flip(img, xy, -1);

	Mat mulx, muly;
	//ƴ��
	hconcat(img, img, mulx);//����ƴ�� ��Ҫ��ͬ

	vconcat(img, img, muly);//����ƴ�� ��Ҫ��ͬ
}

void Demo::Demo14_Rotat() {//ƽ�� ���� ��ת 
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat rotation0,dis0;

	//1.��ȡ������� 2*3����
	rotation0=getRotationMatrix2D(Point2f(img.rows/2.0, img.cols/2.0), -45.0, 0.5);//������ת������ 0.5 ������ı�������

	warpAffine(img, dis0, rotation0,img.size());
	imshow("test", dis0);


	//2.֪��ԭʼ�ͷ����3�� ���Լ������任����
	Point2f src_points[3];
	Point2f dis_points[3];
	src_points[0] = Point2f(0, 0);//ԭʼ��
	src_points[1] = Point2f(0, (float)(img.cols-1));
	src_points[2] = Point2f((float)(img.rows - 1), (float)(img.cols - 1));

	dis_points[0] = Point2f((float)(img.rows - 1)*0.11, (float)(img.cols - 1)*0.20);//�任��ĵ�
	dis_points[1] = Point2f((float)(img.rows - 1)*0.15, (float)(img.cols - 1)*0.70);
	dis_points[2] = Point2f((float)(img.rows - 1)*0.81, (float)(img.cols - 1)*0.85);


	Mat rotation1, dis1;
	rotation1=getAffineTransform(src_points,dis_points);
	warpAffine(img, dis1, rotation1, img.size());

}

void Demo::Demo15_TouShi() {//͸�ӱ任���ĵ�任��  �任����3*3
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);

	Point2f src_points[4];
	Point2f dis_points[4];

	Mat rotation0, dis0;
	rotation0=getPerspectiveTransform(src_points, dis_points);//��ȡ͸�ӱ任����

	warpPerspective(img,dis0,rotation0,img.size());//͸�ӱ任����
}

void Demo::Demo16_Draw() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	line(img, Point(0, 0), Point(200, 200), Scalar(255, 0, 0), 3, LINE_AA, 0);//���һ������ʵ�ֵ��΢ƫ��  ��Ϊ�������� 
	circle(img, Point(50, 50), 50, -1, 8, 0);
	ellipse(img, Point(100, 100), Size(50, 100),-45,0,180,Scalar(255),-1,8,0);//Size �������ľ��볤����Ͷ̰���Ĵ�С 0 180Ϊ��Բ�ĽǶ�
	rectangle(img, Point(100, 100), Point(200, 200), Scalar(100), -1, 8, 0);


	Mat temp = img.clone();
	Point pp[2][6];
	Point pp2[5];

	//const Point* pts[3] = { pp[0],pp[1],pp2 };
	//int npts[] = { 6,6,5 };

	////����λ���
	//fillPoly(temp,pts,npts,3,Scalar(255));//�ڶ�������Ϊ��ά����  ÿ�����������һ�������   ����������Ϊÿ��������ĵ�ĸ���  ���ĸ�����Ϊ���ƶ���εĸ���

	//��������
	putText(temp, "test", Point(50, 50),0,2,Scalar(255),5,8,false);//����������Ϊ�������ֵ����½�����  ���һ��������Ϊtrue  ��ͼ��ԭ�������½ǣ�Ĭ��Ϊ���Ͻǣ�
}

void Demo::Demo17_ROI() {
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	
	Mat dis, ROI1, ROI2;
	Rect rect(206, 206, 200, 200);  //��ʼ����  ����
	ROI1 = img(rect);//��ȡ

	ROI2=img(Range(0, 500), Range(0, 300));//��ȡ ע���д

	//dis=img(Rect(Point2i(200, 200), 200, 200));

	//���
	copyTo(img, dis, Mat());//cv����
	img.copyTo(dis, Mat());//Mat����

	//���ǳ������ͼ��Ӧ�õ����ͼ���� ����һ�����޸� ����ȫ��Ҳ����ű��޸�
}

void Demo::Demo18_PyrDown() {//�²�������С   �����������С ����ģ�� ����ʶ��
	Mat img = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	resize(img, img, img.size() / 2, 0, 0);
	//imwrite("F:/test-picture/5.jpg", img);

	vector<Mat> Guass;
	int level = 3;
	Guass.push_back(img);
	for (int i = 0; i < level; i++) {
		Mat temp = Guass[i];//ǰһ��
		Mat guass;
		pyrDown(temp, guass, temp.size() / 2);//���һ������Ϊ ��䷽��
		Guass.push_back(guass);
	}

	for (int i = 0; i < level; i++) {
		String name = to_string(i);
		imshow(name, Guass[i]);
	}
	waitKey(0);
}

void Demo::Demo19_Pyr() {//�в������   ԭͼ - ��ԭͼ�²���+�ϲ����� ��ͼ��   ��bug����������
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat dis;

	vector<Mat> Guass;
	int level = 3;
	Guass.push_back(img);
	for (int i = 0; i < level; i++) {
		Mat temp = Guass[i];//ǰһ��
		Mat guass;
		pyrDown(temp, guass, temp.size() / 2);//���һ������Ϊ ��䷽��
		Guass.push_back(guass);
	}

	vector<Mat> Lap;
	for (int i = level-1; i > 0; i--) {
		Mat lap, up;
		if (i == level - 1) {
			Mat down;
			pyrDown(Guass[i], down, Guass[i].size() / 2);//����С
			pyrUp(down, up, down.size() * 2);//��Ŵ�
			lap = Guass[i] - up;//ԭͼ-
		}
		else {//ԭ���Ľ���������С��ͼ�� ////����������
			pyrUp(Guass[i], up, Guass[i].size() * 2);//�Ŵ�
			lap = Guass[i - 1] - up;
		}
		Lap.push_back(lap);
	}
}

Mat img; //�ص�����
void callBack(int value,void *) {//��һ�������� createTrackbar�е�value
	float a = value / 100;
	Mat img2 = img * a;
	imshow("img", img2);//��Ҫ��ʾ��ԭ������
}

void Demo::Demo20_Bar() {
	img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	namedWindow("img", WINDOW_AUTOSIZE);
	imshow("img", img);

	int value = 50;
	createTrackbar("�ٷֱ�", "img", &value, 655, callBack, 0);//��ʼֵ ���ֵ 
	waitKey(0);
}

Mat imgPoint;
Point prePoint;
void Mouse(int event, int x, int y, int flags, void* userdata) {//�ص����� envent ����ʱ��������¼�λ  flags ��ⳤʱ����� ��ק ���̰���
	if (event == EVENT_RBUTTONDOWN) {
		cout << "������ܻ���" << endl;
	}
	if (event == EVENT_LBUTTONDOWN) {
		prePoint = Point(x, y);//��¼��ʼ����
		cout << "��ʼ����" << prePoint << endl;
	}if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {//��flags��EVENT_FLAG_LBUTTON���а�λ�����㣬��������Ϊ0�����ʾ���������
		Point pt(x, y);
		line(img, prePoint, pt, Scalar(255), 2, 8, 0);
		prePoint = pt;
		imshow("ͼ1", img);
	}
}

void Demo::Demo21_Mouse() {
	img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	img.copyTo(imgPoint);
	imshow("ͼ1", img);
	imshow("ͼ2", imgPoint);

	setMouseCallback("ͼ1", Mouse, 0);
	waitKey(0);
}

//����ֱ��ͼ
void drawHist(Mat& hist, int type, string name) {//��һ��������ֱ��ͼ
	int hist_w = 512;
	int hist_h = 400;
	int width = 2;
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);//���û���

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

	Mat hist;//���ֱ��ͼͳ�ƽ��
	int channels[1] = { 0 };//ͨ������
	int bins[1] = { 256 };//�Ҷ�ͼ�����ֵ
	const float inRanges[2] = { 0,255 };//�Ҷ�ֵ�仯��Χ
	const float* ranges[1] = { inRanges };//ÿ��ͨ���Ҷ�ֵȡֵ��Χ �����һ��ͨ��  �ǵü�const������

	calcHist(&gray, 1, channels, Mat(), hist, 1, bins, ranges);// ͼ�� ͼ������� ͨ������������ά���飩 mask ������� ÿ��ά��ֱ��ͼ������ߴ磨�������ֵ��  
}

void Demo::Demo23_CalcHist() {//�����طֲ����ӹ㷺
	img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat equalImg;
	equalizeHist(gray, equalImg);//ֱ��ͼ���⻯

	Mat hist1,hist2;//���ֱ��ͼͳ�ƽ��
	int channels[1] = { 0 };//ͨ������
	int bins[1] = { 256 };//�Ҷ�ͼ�����ֵ
	const float inRanges[2] = { 0,255 };//�Ҷ�ֵ�仯��Χ
	const float* ranges[1] = { inRanges };//ÿ��ͨ���Ҷ�ֵȡֵ��Χ �����һ��ͨ��  �ǵü�const������

	calcHist(&gray, 1, channels, Mat(), hist1, 1, bins, ranges);
	calcHist(&equalImg, 1, channels, Mat(), hist2, 1, bins, ranges);

	drawHist(hist1, NORM_INF, "hist1");
	drawHist(hist2, NORM_INF, "hist2");

	imshow("ԭͼ",gray);
	imshow("���⻯", equalImg);

	waitKey(0);
}

void Demo::Demo24_Hist() {//ֱ��ͼƥ�� ��ԭͼ��ֱ��ͼ���ۼƸ�����Ŀ��ֱ��ͼ���ۼƸ��ʱȽ�  ����ӳ���ϵ       ��bug������
	Mat img1 = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat img2 = imread("F:/test-picture/6.jpg", IMREAD_COLOR);

	Mat hist1, hist2;
	int channels[1] = { 0 };//ͨ������
	int bins[1] = { 256 };//�Ҷ�ͼ�����ֵ
	const float inRanges[2] = { 0,255 };//�Ҷ�ֵ�仯��Χ
	const float* ranges[1] = { inRanges };//ÿ��ͨ���Ҷ�ֵȡֵ��Χ �����һ��ͨ��  �ǵü�const������

	calcHist(&img1, 1, channels, Mat(), hist1, 1, bins, ranges);
	calcHist(&img2, 1, channels, Mat(), hist2, 1, bins, ranges);

	drawHist(hist1, NORM_INF, "hist1");
	drawHist(hist2, NORM_INF, "hist2");

	//��������ͼ��ֱ��ͼ���ۼƸ���
	float hist1_cdf[256] = { hist1.at<float>(0) };
	float hist2_cdf[256] = { hist2.at<float>(0) };

	for (int i = 1; i <= 255; i++) {
		hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);//�ۼ�
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

	//�����ۻ�����������
	float diff_cdf[256][256];
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			diff_cdf[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);//Ϊ�˷�������ҳ���С��
		}
	}

	//for (int i = 0; i < 256; i++) {//test
	//	for (int j = 0; j < 256; j++) {
	//		cout<<diff_cdf[i][j];
	//	}
	//	cout << endl;
	//}
	//cout << endl;

	//����LUTӳ���
	Mat lut(1, 256, CV_8UC1);

	for (int i = 0; i < 256; i++) {
		//���һҶȼ�Ϊi��ӳ��Ҷ�
		//��i���ۼƸ��ʲ���С�Ĺ涨Ϊ�Ҷ�
		float min = diff_cdf[i][0];
		int index = 0;//��¼��С���±�(Ҳ�ǻҶ�)
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
	imshow("ԭͼ", img1);
	imshow("ƥ��ģ��", img2);
	imshow("ƥ����", result);

	calcHist(&result, 1, channels, Mat(), hist3, 1, bins, ranges);
	drawHist(hist3, NORM_INF, "hist3");
	waitKey(0);
}

void Demo::Demo25_Match() {//ģ��ƥ��
	Mat img1 = imread("F:/test-picture/1.jpg", IMREAD_COLOR);
	Mat img2 = imread("F:/test-picture/6.jpg", IMREAD_COLOR);
	Mat dis;
	matchTemplate(img1,img2,dis,TM_CCOEFF_NORMED,Mat());//����Ϊcv_8u  cv_32f ���Ϊcv32f

	double min;
	double max;
	Point minp;
	Point maxp;
	minMaxLoc(dis, &min, &max, &minp, &maxp);//�����ͼ����Ѱ������ֵ

	rectangle(img1, Point(maxp.x,maxp.y), Point(maxp.x + img2.cols, maxp.y + img2.rows),Scalar(255),5,8,0);
	waitKey(0);
}

void Demo::Demo26_filter() {
	Mat img1 = imread("F:/test-picture/5.jpg", IMREAD_COLOR);
	Mat kernel = (Mat_<float>(3, 3) << 1, 2, 1,
		2, 0, 2,
		1, 2, 1);
	Mat kernel_norm = kernel / 12;//���ģ���һ�� ��ֹ��������255

	Mat result, result_norm;
	filter2D(img1, result, CV_32F, kernel, Point(-1, -1), 2, 4);//�������������ͬ�ĳߴ��ͨ��  ������������ͻᷢ���ı�  �����cv32fc1  anchor��׼�㣨һ��Ϊ���������λ�ã� ƫ����  �������
	filter2D(img1, result_norm, CV_32F, kernel_norm, Point(-1, -1), 2, 4);
	imshow("result_norm", result_norm);//��ʾ����������
	waitKey(0);
}

void Demo::Demo27_Noise() {
	Mat img1 = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	//����������Ҫ�ֶ���� ������ͨ�� ��ÿ��ͨ����Ҫ���
	cvflann::rand_double();
	cvflann::rand_int();

	//��Ӹ�˹����ǰҪ�ȴ���һ����ͼ��ߴ� �������ͺ�ͨ��������ͬ��mat�����
	Mat noise = Mat::zeros(img1.size(), img1.type());
	//RNG::fill()
	RNG rng;//Ҫ��ʵ����
	rng.fill(noise, RNG::NORMAL, 10, 20);//��ֵ�ͷ���  ��������

	img1 = img1 + noise;//�������
	imshow("111", img1);
	//imwrite("F:/test-picture/gaosi.jpg", img1);
	waitKey(0);
}

void Demo::Demo28_LinearFilter() {//�����˲�
	//��ֵ�˲� 
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat dis1,dis2,dis3;
	blur(img1, dis1, Size(3, 3), Point(-1, -1));//�ú���ǿ�ƽ��й�һ��

	//�����˲�
	boxFilter(img1,dis3,-1,Size(3,3),Point(-1,-1),false);//�;�ֵһ�� ����ѡ���Ƿ��һ�� false �����й�һ��

	//��˹�˲�0
	GaussianBlur(img1, dis2, Size(15, 15), 0, 0);//���size����Ϊ0 ���ɱ�׼�����ߴ�  x �� y������˲�����׼ƫ�� ���Ϊ0 ����Size������
	waitKey(0);
}

void Demo::Demo29_MidFilter() {//��ֵ�˲� ��Чȥ����������
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat dis;

	medianBlur(img1, dis, 7);
	waitKey(0);
}

void Demo::Demo30_SepFilter() {
	//�����˲����пɷ�����  �ȶ�x�ٶ�y  =  ��x��y  �ܼ��ټ�����
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat dis1,dis2,dis3,dis4,dis5;
	Mat a = (Mat_<float>(3, 1) << -1, 3, -1);
	Mat b = a.reshape(1,1);
	//cn��ͨ������ָ��Mat�������ͨ���������Ϊ0�����ʾ����ԭͨ�������䡣
	//rows��������ָ��Mat�������������
	Mat ab = a * b;

	filter2D(img1, dis1, -1, a, Point(-1, -1), 0);//-1��ʾ���ͼ������������ͼ����ͬ
	filter2D(dis1, dis2, -1, b, Point(-1, -1), 0);

	filter2D(img1, dis3, -1, ab, Point(-1, -1), 0);//��������������һ��  ˳�򲻽���

	sepFilter2D(img1, dis4 ,-1,a ,Mat(),Point(-1,-1),0);//���������������޸�ͼ����������� x�����˲��� y�����˲���
	sepFilter2D(img1, dis4, -1, Mat(), b, Point(-1, -1), 0);

	Mat gauss = getGaussianKernel(3,1);//�ߴ��������ֵ ��ȡ��˹�˲���
}

void Demo::Demo31_Edge1() {//��Ե��� һ�׵������ֵ���仯��� f��x+1,y��-f(x-1,y)/2 
	//Sobel
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	Mat disx,disy,disxy;


	//һ�����м�� ���м��
	Sobel(img1, disx, CV_16S,1,0,3);//CV_16S ��Ϊ������ܻ���ָ���   dx dy��ʾ��x��y���󵼵Ľ��� 
	convertScaleAbs(disx, disx);//�Ա�Ե�������ȡ����ֵ

	Sobel(img1, disy, CV_16S, 0, 1, 1);//ksize ����1��3 ��һ�µ�
	convertScaleAbs(disy, disy);
	disxy = disx + disy;//��б�ı�Ե���ܻᱻ��ǿ
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

	//��ȡһ��x�����sobel����  ???
	getDerivKernels(sobelx, sobely,1,0,3);//3*1
	cout << sobelx << endl;
	cout << sobely << endl;
	sobelx.reshape(CV_8U, 1);//1*3
	sobelX = sobely * sobelx;
	cout << sobelX << endl;

	//��ȡscharr����
	getDerivKernels(scharrx, scharry, 1, 0, FILTER_SCHARR);
	scharrx.reshape(CV_8U, 1);//1*3
	scharrX = scharry * scharrx;
}

void Demo::Demo32_Edge2() {
	//Laplacian �޹ط�����ȡ��Ե �����ܵ�����Ӱ�� 
	Mat img1 = imread("F:/test-picture/gaosi.jpg", IMREAD_COLOR);
	cvtColor(img1, img1, COLOR_BGR2GRAY);

	Mat dis1, dis2, dis3;
	Laplacian(img1, dis1, CV_16S , 3);
	convertScaleAbs(dis1, dis1);//���и�ֵ���ص�ͼ��ת��Ϊ�޷����������ͣ���ȡ����ֵ����������
	
	//Canny �ܹ�ȥ����ٱ�Ե�����յ����� 1.��˹ƽ�� 2.�ݶȼ�������ͷ�ֵ 3.�Ǽ���ֵ���� 4.˫��ֵ����ǿ ����Ե 5.������������Ե
	Canny(img1, dis2,100,200 ,3);// ������ֵ sobel����ֱ��  ����ǵ�ͨ��ͼ�� ��ֻ��0��255����ֵcv_8u    

	//���˲���Canny
	GaussianBlur(img1, dis3, Size(3, 3), 0, 0);
	Canny(dis3, dis3, 100, 200, 3);
	waitKey(0);
}

void Demo::Demo33_Connect() {//��ͨ��ָ�
	//һ���ȶ�ֵ������
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat rice,riceBW;
	cvtColor(img1, rice, COLOR_BGR2GRAY);//��ֵ��֮ǰ�ȱ�ɻҶ�ͼ��
	threshold(rice, riceBW, 50, 255, THRESH_BINARY);

	RNG rng(1);
	Mat out;
	int num=connectedComponents(riceBW, out ,8,CV_16U);//����Ϊcv_8u  connectivity8Ϊʹ��8��ͨ����  ͳ��ͼ������ͨ��ĸ���  bug
	//����еĲ�ͬ������ֵ��ʾ���ǲ�ͬ����ͨ��  num���Ƿ������ֵ�ĸ���

	vector<Vec3b> colors;//��ÿ��һ��ͨ������һ����ɫ
	for (int i = 0; i < num; i++) {
		Vec3b vec3 = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		colors.push_back(vec3);
	}

	Mat result = Mat::zeros(img1.size(), img1.type());//��ɫͨ��Ϊ3 ��ɫͨ��ҲΪ3
	int w = result.cols;
	int h = result.rows;
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			int label = out.at<uint16_t>(r, c);
			if (label == 0) {//��ɫ�Ĳ��ı�
				continue;
			}
			result.at<Vec3b>(r, c) = colors[label];//ͬһ����ֵ ��ʾͬһ����ɫ
		}
	}

	imshow("ԭͼ", img1);
	imshow("��Ǻ�ͼ��", result);
	waitKey(0);


	//����ͳ����ͨ����Ϣ
	Mat stats, centroids;
	num=connectedComponentsWithStats(riceBW,out,stats,centroids,8,CV_16U);//stats ��ͬ��ͨ���ͳ����Ϣ���� centroids ÿ����ͨ�����������  
	vector<Vec3b> new_colors;//��ÿ��һ��ͨ������һ����ɫ
	for (int i = 0; i < num; i++) {
		Vec3b vec3 = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		new_colors.push_back(vec3);
	}

	for (int i = 0; i < num; i++) {
		//����λ��
		int cen_x = centroids.at<double>(i, 0);//��ȡÿ����ͨ�����������
		int cen_y = centroids.at<double>(i, 1);

		//���α߿�
		int x = stats.at<int>(i, CC_STAT_LEFT);//�����
		int y = stats.at<int>(i, CC_STAT_TOP);//����
		int w = stats.at<int>(i, CC_STAT_WIDTH);//��
		int h = stats.at<int>(i, CC_STAT_HEIGHT);//��

		//�����ĵ�
		circle(img1, Point(cen_x, cen_y), 2, Scalar(0, 255, 0), 2);

		//��Ӿ���
		Rect rect(x, y, w, h);
		rectangle(img1, rect, new_colors[i], 1);
		putText(img1, format("%d", i), Point(cen_x, cen_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
	}
	imshow("��Ǻ�", img1);
}

void Demo::Demo34_DisTran() {//����ͼ����ÿ����0���ؾ������0����(�߽�)�ľ���
	Mat a = (Mat_<uchar>(5, 5) << 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 0, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1);
	Mat dist_L1, dist_L2, dist_C, dist_L12;

	distanceTransform(a,dist_L1,1,3,CV_8U);//����ҪΪcv_8u�ĵ�ͨ��ͼ�� �������Ϊcv_8u ����cv_32f 1 �������� 2ŷʽ���� 3���̾���
	cout << dist_L1 << endl;

	distanceTransform(a, dist_L2, 2, 3, CV_8U);//5��ʾ����ߴ�Ϊ5x5 //
	cout << dist_L2 << endl;

	distanceTransform(a, dist_C, 3, 3, CV_8U);//5��ʾ����ߴ�Ϊ5x5
	cout << dist_C << endl;

	//waitKey(0);

	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat rice, riceBW, riceBW_INV;
	cvtColor(img1, rice, COLOR_BGR2GRAY);//��ֵ��֮ǰ�ȱ�ɻҶ�ͼ��
	threshold(rice, riceBW, 50, 255, THRESH_BINARY);
	threshold(rice, riceBW_INV, 50, 255, THRESH_BINARY_INV);

	//����任
	Mat dist, dist_INV;
	distanceTransform(riceBW, dist, 1, 3, CV_32F);
	distanceTransform(riceBW_INV, dist_INV, 1, 3, CV_8U);

	waitKey(0);
}

void Demo::Demo35_Erode() {//��̬ѧ���������ڶԶ�ֵ��ͼ����� ��ʴ������ȥ��΢С���� �ָ�ճ������
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);

	//���ɽṹԪ�� //���ڸ�ʴ/���͵ľ����
	Mat s1=getStructuringElement(0,Size(3,3),Point(-1,-1));//���� ��С ���ĵ�
	Mat s2 = getStructuringElement(1, Size(3, 3));
	//0���� 1ʮ���� 2��Բ
	
	Mat dis1,dis2;
	erode(img1,dis1,s1);//itera ��ʴ����
	erode(img1, dis2, s2,Point(-1,-1),10);

	waitKey(0);
}

void Demo::Demo36_Dilate() {//����
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);

	//���ɽṹԪ��
	Mat s1 = getStructuringElement(0, Size(3, 3));//���� ��С ���ĵ�
	Mat s2 = getStructuringElement(1, Size(3, 3));

	Mat dis1, dis2;
	dilate(img1, dis1, s1, Point(-1, -1), 5);
	dilate(img1, dis2, s2, Point(-1, -1), 10);

	waitKey(0);
}

void Demo::Demo37_Morpho() {
	//������ �ȸ�ʴ������ ������С��ͨ��������  �����ϴ���ͨ���ٴ����ͣ�

	//������ �����ͺ�ʴ  ���С�ն� ���������ٽ���ͨ��

	//��̬ѧ�ݶ� ԭͼ������-ԭͼ��ʴ  ͻ��ͼ���еı�Ե�����������

	//��ñ �õ�ԭͼ���з�ɢ��һЩ�ߵ� ԭͼ-ԭͼ������  ͻ��ͼ���б���Χ������Ŀ��

	//��ñ �õ�ԭͼ���а�һ�������  ԭͼ������-ԭͼ  ͻ��ͼ���б���Χ������Ŀ��

	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	threshold(img1, img1, 80, 255, THRESH_BINARY);//��̬ѡ�����ڶ�ֵ����ͼ��
	Mat dis0,dis1, dis2, dis3, dis4, dis5, dis6,dis7;

	//��ȡ�ṹԪ��
	Mat k = getStructuringElement(1, Size(3, 3));
	morphologyEx(img1,dis0,0,k);//0��ʴ 1���� 2�� 3�� 4�ݶ� 5��ñ 6��ñ 7���� 
	morphologyEx(img1, dis1, 1, k);
	morphologyEx(img1, dis2, 2, k);
	morphologyEx(img1, dis3, 3, k);
	morphologyEx(img1, dis4, 4, k);
	morphologyEx(img1, dis5, 5, k);
	morphologyEx(img1, dis6, 6, k);
	//morphologyEx(img1, dis7, 7, k);

	waitKey(0);
}

void Demo::Demo38_ximgproc() {//ͼ��ϸ�����Ǽܻ���  һ���Ƕ�ֵ��ͼ�� �Ҷ�Ҳ����   ��չģ��û��װ�á����������ˡ�����
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	threshold(img1, img1, 80, 255, THRESH_BINARY);//��̬ѡ�����ڶ�ֵ����ͼ��

	//ximgproc
}

void Demo::Demo39_Contours() {//����
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat binary,gray;
	cvtColor(img1, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(13, 13), 4, 4);
	threshold(gray, binary, 50, 1, THRESH_BINARY);//��̬ѡ�����ڶ�ֵ����ͼ��  С��50����0 ���ڵ���255  
	//����Ϊ��ͨ�����߶�ֵͼ��
	//cout << binary.type() << endl;

	vector<vector<Point>> contours;//���� �洢��⵽���������������� �洢��⵽����������������
	vector<Vec4i> hierarchy;//�洢�����Ĳ�νṹ��Ϣ
	findContours(binary, contours, hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());//�ڶ�������Ϊ��⵽������ ��������ص�����  bug???***********************  finish������

	for (int i = 0; i < hierarchy.size(); i++) {
		cout << hierarchy[i] << endl;
	}

	//��������
	for (int t = 0; t < contours.size(); t++) {
		drawContours(img1, contours, t, Scalar(0, 0, 255), 2, 8);//t��ָ��Ҫ���Ƶ����������� ����-1 ������������
		imshow("111", img1);
		waitKey(0);
	}
}

void Demo::Demo40_AreaAndLength() {
	Mat img1 = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	Mat binary, gray;
	cvtColor(img1, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(13, 13), 4, 4);
	threshold(gray, binary, 50, 1, THRESH_BINARY);//��̬ѡ�����ڶ�ֵ����ͼ��  С��50����0 ���ڵ���255  

	vector<vector<Point>> contours;//���� �洢��⵽���������������� �洢��⵽����������������
	vector<Vec4i> hierarchy;//�洢�����Ĳ�νṹ��Ϣ
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

	//����������
	for (int i = 0; i < contours.size(); i++) {
		double area1 = contourArea(contours[i]);//����ֵΪdouble
		cout << area1 << endl;
	}

	//�����������
	for (int i = 0; i < contours.size(); i++) {
		double length1 = arcLength(contours[i],true);//����ֵΪdouble true��ʾ����������Ǳպϵ�
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
	Mat img1, img2;//1������ 2��С���
	img.copyTo(img1);
	img.copyTo(img2);

	Mat canny;
	Canny(img, canny, 80, 160, 3, false);//����Ե

	Mat kernel = getStructuringElement(0, Size(3, 3), Point(-1, -1));
	dilate(canny, canny, kernel);//���� ��ϸС��϶�
	imshow("canny", canny);

	vector<vector<Point>> contours;//����
	vector<Vec4i> hierarchy;//�����Ϣ
	findContours(canny, contours, hierarchy, 0, 2, Point());

	//boundingRect();//����ֵΪ���� �õ������Ӿ��� ����Ϊvecctor<Point>(һ������) ����Mat
	//minAreaRect();//����ֵ����ת����RotateRect  
	//approxPolyDP();//��������
	for (int i = 0; i < contours.size(); i++) {
		Rect rect = boundingRect(contours[i]);//��ȡ�����Ӿ���
		rectangle(img1, rect, Scalar(0, 0, 255), 2, 8, 0);

		RotatedRect rrect = minAreaRect(contours[i]);//��С��Ӿ���  ��������RotatedRect���Է����ĸ�����(Point2f����)�����ĵ�
		Point2f points[4];
		rrect.points(points);//rrect.points() �����Ĳ�����һ�� OutputArray ���������ڽ��սǵ�����
		Point2f cpt = rrect.center;//��ȡ����

		//������ת����������λ��
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

	cout << "��������" << endl;
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

		Mat result;//n*2  ������vector<Point>����
		approxPolyDP(contours2[i], result, 4, true);//�������� result����ʾ����Ľ��ƶ���εĵ�����  4����ʾ���ƶ���εľ��Ȳ��� ԽСԽ�ӽ�ԭʼ����  true��ʾ�պ�
		drawapp(result, approx);//��������
		cout << result.rows << endl;//��ĸ���

		if (result.rows == 3) {
			putText(approx, "triangle", center ,0, 1, Scalar(0, 255, 0), 1, 8);
		}
		else if (result.rows == 4) {
			putText(approx, "rectangle", center, 0, 1, Scalar(0, 255, 0), 1, 8);
		}
	}
	imshow("result", approx);
}

void Demo::Demo42_Hull() {//͹�����
	Mat img = imread("F:/test-picture/handb.jpg");
	Mat gray, binary;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 50, 255, THRESH_BINARY);//��ֵ̫��ᵼ����������

	//��������������
	Mat k = getStructuringElement(0, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary,2, k, Point(-1, -1));
	imshow("binay", binary);

	//��������
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	findContours(binary, contours, 0, 2, Point());

	for (int n = 0; n < contours.size(); n++) {
		vector<Point> hull;//͹��
		//��������㼯����������͹��
		convexHull(contours[n], hull);
		for (int i = 0; i < hull.size(); i++) {
			circle(img, hull[i], 4, Scalar(255, 0, 0), 2, 8, 0);//��ԭͼ�ϻ���
			if (i == hull.size() - 1) {
				line(img, hull[i], hull[0], Scalar(0, 0, 255), 2, 8, 0);
				break;//ֱ��break ������һ�б���
			}
			line(img, hull[i], hull[i + 1], Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	imshow("hull", img);
	waitKey(0);
}

void drawLine(Mat& img, vector<Vec2f> lines, double rows, double cols, Scalar scalar, int n) {//ԭͼ��� ԭͼ��� ��ɫ �߿�
	//lines�������Ǽн��Լ�ֱ�߾�������ԭ��ľ���
	Point pt1, pt2;//���ݼҽ��Լ�����ԭ����� �ȼ����������ص�
	for (int i=0; i < lines.size(); i++) {
		float rho = lines[i][0];//ֱ�߾�������ԭ��ľ���
		float theta = lines[i][1];//ֱ�߹�����ԭ�㴹����x��н�
		double co = cos(theta);
		double si = sin(theta);

		double x0 = rho * co;//����
		double y0 = rho * si;
		double length = max(rows, cols);

		pt1.x = cvRound(x0 + length*(-si)); //cvRound�Ը�����������������ȡ��
		pt1.y = cvRound(y0 + length * (co));

		pt2.x = cvRound(x0 - length * (-si));//???
		pt2.y = cvRound(y0 - length * (co));

		line(img, pt1, pt2, scalar, n);
	}
}

void Demo::Demo43_HoughLine() {//ֱ�߼�� ԭ�ռ�һ���㣨���Զ����ߴ����� -> �����ռ�һ���ߣ����е�ÿ���� ����ԭ�ռ�����һ��ֱ�ߣ�
							//									����� -> ������ ��������ֱ���ཻ �����Ӧ��ֱ�� ��Ӧ��ԭ�ռ��й������ֱ��
							//									���������ֱ��  -> ÿ�����Ӧֱ���ཻ��һ�㣩
							//									ԭ�ռ�ֱ�߾����ĵ�=�����ռ佻�㱻ֱ�߾����Ĵ�����������ֵ Ѱ��ֱ�ߣ�

	//1.�����ռ���ɢ�� 2.ӳ�� 3.ͳ��ÿ��������ֵĴ��� ѡȡ����ĳһ��ֵ�ķ�����Ϊ��ʾֱ�ߵķ��� 4.�������ռ��ʾֱ�ߵķ��������Ϊͼ����ֱ�ߵĲ���
	Mat img = imread("F:/test-picture/box.jpg", IMREAD_GRAYSCALE);
	Mat edge;
	Canny(img, edge, 80, 180, 3, false);
	threshold(edge, edge, 170, 255, THRESH_BINARY);//�ȱ�Ե��� ��ֱ�߼��

	vector<Vec2f> lines1, lines2;
	vector<Vec4i> linesP3, linesP4;
	HoughLines(edge, lines1, 1, CV_PI / 180, 50,  0, 0);//1��ɢ���ĵ�λ  50��ֵֻ�дﵽ����Ϊ��ֱ�� 
	HoughLines(edge, lines2, 1, CV_PI / 180, 150, 0, 0);

	HoughLinesP(edge, linesP3, 1, CV_PI / 180, 150, 30, 10);//150 ��ֵ��Ӱ�������� 30�����߶γ��ȣ�С����ȥ  10�߶μ�������������Ӱ�쳤�̣�
	HoughLinesP(edge, linesP4, 1, CV_PI / 180, 150, 30, 30);//�õ��Ľ��linesP4��������  

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

void Demo::Demo44_Enclose() {//����ɢ��������
	//Vec4f lines;
	//vector<Point2f> points;

	//double parm = 0;
	//double reps = 0.01;//����ԭ����ֱ��֮��ľ��뾫��
	//double aeps = 0.01;//�ǶȾ���

	////ֱ����� 2d����3d�㼯  2d�㼯��������Ϊvec4f 3dΪvec6f 
	//fitLine(points, lines, DIST_L1, 0, reps, aeps);//parm:��� distType Ϊ CV_DIST_L2����ò���Ϊ 0����� distType Ϊ CV_DIST_L1����ò���Ϊ 0.01
	////lines (vx, vy, x0, y0)  ��������(vx, vy)��ʾֱ�ߵķ���(x��y�������)�� (x0, y0) ���ʾֱ���ϵ�һ���㡣
	//double k = lines[1] / lines[0];//б��

	//waitKey(0);

	Mat img(500, 500, CV_8UC3, Scalar::all(0));//Scalar::all(0) ����һ�����з�����Ϊ 0 �� Scalar ����
	RNG& rng = theRNG();//// ��ȡĬ�ϵ����������������  �����RNG rng ���ַ����й���

	while (true)
	{
		int i, count = rng.uniform(1, 101);
		vector<Point> points;
		//���������
		for (int i = 0; i < count; i++) {
			Point pt;
			pt.x = rng.uniform(img.cols / 4, img.cols * 3 / 4);
			pt.y = rng.uniform(img.rows / 4, img.rows * 3 / 4);
			points.push_back(pt);
		}

		//���������
		vector<Point2f> triangle;
		double area = minEnclosingTriangle(points, triangle);//����ֵ��ʾ�ҵ�����С�����ε����

		//���Բ
		Point2f center;
		float radius = 0;
		minEnclosingCircle(points, center, radius);

		//������ͼ���������
		img = Scalar(0);//��0
		Mat img2;
		img.copyTo(img2);

		//���Ƶ�
		for (int i = 0; i < count; i++) {
			circle(img, points[i], 3, Scalar(255, 255, 255), FILLED, LINE_AA);
			circle(img2, points[i], 3, Scalar(255, 255, 255), FILLED, LINE_AA);
		}

		//����������
		for (int i = 0; i < 3; i++) {
			if (i == 2) {
				line(img, triangle[i], triangle[0], Scalar(255, 255, 0), 1, 16);
				break;
			}
			line(img, triangle[i], triangle[i+1], Scalar(255, 255, 0), 1, 16);
		}

		//����Բ��
		circle(img2, center, radius, Scalar(255, 255, 0), 1, LINE_AA);

		imshow("triangle", img);
		imshow("circle", img2);

		char key = (char)waitKey();
		if (key == 27 || key == 'q') {
			break;
		}
	}
}

void Demo::Demo45_Code() {//��ά��ʶ��
	Mat img = imread("F:/test-picture/code.jpg", IMREAD_COLOR);
	Mat gray, qrcode;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	QRCodeDetector qrcodee;//��ʵ����
	vector<Point> points;
	string info;
	bool isQRcode;
	isQRcode = qrcodee.detect(gray, points);//��ȡ��ά��λ����Ϣ

	if (isQRcode) {
		info = qrcodee.decode(gray, points, qrcode);//�����ά�� qrcode��ȡ��ά��
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

	//��λ����������
	string info2;
	vector<Point> points2;
	info2 = qrcodee.detectAndDecode(gray, points2);
	cout << points2 << endl;

	putText(img, info2.c_str(), Point(20, 55), 0, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("img", img);
	imshow("qrcode", qrcode);
}

void Demo::Demo46_Integral() {//����ͼ�� ����ͼ�����ش�����̣���������ǰ���
	Mat img = Mat::ones(16, 16, CV_32FC1);

	RNG rng(22);
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			float d = rng.uniform(-0.5, 0.5);
			img.at<float>(y, x) = img.at<float>(y, x) + d;
		}
	}

	Mat sum,sqsum,tilted;
	integral(img, sum,sqsum,tilted);//��׼��� ƽ����� ���������
	Mat sum8U = Mat_<uchar>(sum);
	Mat sqsum8U = Mat_<uchar>(sqsum);//
	Mat tilted8U = Mat_<uchar>(tilted);//bug ??? finish

	waitKey(0);
}

void Demo::Demo47_floodFill() {//��ˮ��� 1.ѡ���ӵ� 2.�ж�4����8����������ֵ������ֵ��ֵ С����ֵ����ӽ����� 3.�¼ӵ����ص���Ϊ�µ����ӵ�
	Mat img = imread("F:/test-picture/rice.jpg", IMREAD_COLOR);
	RNG rng(22);

	int connectivity = 4;//��ͨ����ʽ
	int maskVal = 255;//����ͼ����ֵ  ����ͼ�����ڱ���ѷ��ʵ����ء�255 ��һ��������ѡ���ʾ������ͼ������������������Ϊ���ֵ
	int flags = connectivity | (maskVal << 8) | FLOODFILL_FIXED_RANGE;//��������ʽ��־

	//������ѡ�����ص�Ĳ�ֵ  ���¶�����20
	Scalar loDiff = Scalar(20, 20, 20);
	Scalar upDiff = Scalar(20, 20, 20);

	Mat mask = Mat::zeros(img.rows + 2, img.cols+2, CV_8UC1);//���ڼ�¼��Щλ�ñ�����

	while (true)
	{
		int py = rng.uniform(0, img.rows - 1);
		int px = rng.uniform(0, img.cols - 1);
		Point point = Point(px, py);

		Scalar newVal = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		int area = floodFill(img, mask, point, newVal,nullptr, loDiff, upDiff, flags); //newVal���ֵ Rect�����������ı߽���� 

		cout << point.x << " " << point.y << " " << area << endl;
		imshow("img", img);

		int c = waitKey(0);
	}
}

void Demo::Demo48_Watershed() {//��ˮ�� 1.���� �Ծֲ���Сעˮ 2.��û Ѱ�һ㼯���� ���Ƿָ���
	Mat img, imgGray, imgMask, img_;//imgMask �ָ���
	Mat maskWaterShed;//Watershed��������
	img = imread("F:/test-picture/wline.png");//���б��ͼ��
	img_ = imread("F:/test-picture/5.jpg");
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	////��ֵ����������
	threshold(imgGray, imgMask, 254, 255, THRESH_BINARY);
	Mat k = getStructuringElement(0, Size(3, 3));
	morphologyEx(imgMask, imgMask, 1, k);

	imshow("���ͼ��", img);
	imshow("ԭͼ��", img_);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgMask, contours, hierarchy, 0, 2);

	//markers �����þ�������Ԥ�Ȱ�һЩ�����ע�ã���Щ��ע�˵������֮Ϊ���ӵ㡣watershed �㷨�����Щ��ǵ��������������������ͼ��
	maskWaterShed = Mat::zeros(img.size(), CV_32S);//��0 1 2
	
	for (int i = 0; i < contours.size(); i++) {
		drawContours(maskWaterShed, contours, i, Scalar::all(i+1), -1, 8, hierarchy, INT_MAX);//��ÿһ������������ȫ�����Ϊ1 2 3...  Ϊʲô������
	}

	RNG rng;
	//��ˮ���㷨 ��ԭͼ����
	watershed(img_, maskWaterShed);//maskҪ��Ҫ��CV_32S
	vector<Vec3b> colors;
	for (int i = 0; i < contours.size(); i++) {
		colors.push_back(Vec3b((uchar)rng.uniform(0, 255), (uchar)rng.uniform(0, 255), (uchar)rng.uniform(0, 255)));
	}

	Mat resultImg = Mat(img.size(), CV_8UC3);//��ʾͼ��
	for (int i = 0; i < imgMask.rows; i++) {
		for (int j = 0; j < imgMask.cols; j++) {
			int index = maskWaterShed.at<int>(i, j);
			if (index == -1) {//�߽�
				resultImg.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else if (index <= 0 || index > contours.size()) {//û�б�ǵ�������0
				resultImg.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else {
				resultImg.at<Vec3b>(i, j) = colors[index - 1];
			}
		}
	}

	imshow("result", resultImg);
	//resultImg = resultImg * 0.8 + img_ * 0.2;//bug ���Ͳ�ƥ�䡣����
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
					resImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);//bug Vec3d������
				}
			}
		}
		//
		imshow(to_string(n), resImage);
	}

	waitKey(0);
}

void Demo::Demo49_Harris() {//Harris�ǵ��� ֻ������Ҷ�ͼ��
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	//����Harrisϵ��
	Mat harris;
	int blockSize = 2;//�ǵ���ʱҪ���ǵ������С�����ֵԽ�󣬼�⵽�Ľǵ�����ҲԽ�ࡣ
	int kSize = 3;//sobel���Ӵ�С
	cornerHarris(gray, harris, blockSize, kSize, 0.04);//k��ȡֵһ��Ϊ0.02-0.04

	Mat harrisn;
	normalize(harris, harrisn, 0, 255, NORM_MINMAX);//��ͼ���һ����0-255
	convertScaleAbs(harrisn, harrisn);//���������ͱ�ΪCV_8U
	//��ͼ��������š������ֵ��ת��Ϊ 8 λ�޷���������ʽ

	//harrisÿ�����ص���ֵ�����˸�λ�ô��Ľǵ���Ӧǿ�ȡ��ǵ���Ӧǿ�ȵ���ֵԽ�󣬱�ʾ��λ��Խ�����ǽǵ㡣

	//Ѱ��harris�ǵ�
	vector<KeyPoint> keyPoints;
	for (int row = 0; row < harrisn.rows; row++) {
		for (int col = 0; col < harrisn.cols; col++) {
			int R = harrisn.at<uchar>(row, col);
			if (R > 208) {
				KeyPoint keyPoint;
				keyPoint.pt.x = row;//ע��.pt
				keyPoint.pt.y = col;
				keyPoints.push_back(keyPoint);
			}
		}
	}

	drawKeypoints(img, keyPoints, img);
	imshow("ϵ������", harrisn);
	imshow("�ǵ�",img);

	waitKey(0);
}

void Demo::Demo50_KeyPoint() {//Shi-Tomas�ǵ�
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	int maxCorners = 100;//���ǵ����Ŀ
	double quality_level = 0.01;//�����ȼ�
	double minDistance = 0.04;//�����ǵ�֮�����Сŷʽ����
	vector<Point2f> corners;
	goodFeaturesToTrack(gray, corners, maxCorners, quality_level, minDistance, Mat(), 3, false);//false ��ʾ���Ǽ��㺣��˼�ǵ�
	//����ֵ���ǽǵ������

	vector<KeyPoint> keyPoints;
	for (int i = 0; i < corners.size(); i++) {
		KeyPoint keyPoint;
		keyPoint.pt = corners[i];
		keyPoints.push_back(keyPoint);
	}

	drawKeypoints(img, keyPoints, img);
	imshow("�ǵ�", img);
	waitKey(0);
}

void Demo::Demo51_SubPix() {//�ǵ�λ���������Ż�

	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	int maxCorners = 100;//���ǵ����Ŀ
	double quality_level = 0.01;//�����ȼ�
	double minDistance = 0.04;//�����ǵ�֮�����Сŷʽ����
	vector<Point2f> corners;
	goodFeaturesToTrack(gray, corners, maxCorners, quality_level, minDistance, Mat(), 3, false);
	//ʹ�� TermCriteria::EPS �� TermCriteria::COUNT �����������а�λ�������ָ����ֹ���������͡�
	//ʹ�õ���ֹ������ TermCriteria::EPS + TermCriteria::COUNT����ʾ��ֹ����ͬʱ�������������Ŀ�꺯��ֵ�ı仯��

	Size winSize = Size(5, 5);
	Size zeroSize = Size(-1, -1);//�������� һ������Ϊ-1 -1
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

void Demo::Demo52_ORB() {//��������� ORB�����㣨������Դ�����٣�
	Mat img = imread("F:/test-picture/5.jpg", IMREAD_COLOR);

	Ptr<ORB> orb = ORB::create(500);//����ʹ��Ĭ��ֵ

	//����ؼ��㣨��û�����ӣ�
	vector<KeyPoint> KeyPoints;
	orb->detect(img, KeyPoints);

	//����ORB������
	Mat description;
	orb->compute(img, KeyPoints, description);

	//����������
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

void Demo::Demo53_FeatureMatch() {//������ƥ��
	Mat img1, img2;
	img1 = imread("F:/test-picture/p1.jpg");
	img2 = imread("F:/test-picture/p2.jpg");

	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;

	//����������
	orb_features(img1, kp1, des1);
	orb_features(img2, kp2, des2);

	vector<DMatch> matches;//���ƥ�����ı���
	BFMatcher matcher(NORM_HAMMING);//����������ƥ���� ʹ�ú���������Ϊ����������֮��Ĳ�������������ڶ�ֵ��������������ORB��BRIEF�ȣ�  ����ƥ��
	matcher.match(des1,des2,matches);//������ƥ��

	cout << "matches" << matches.size() << endl;

	//�ҵ�����������С�����ֵ
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
		if (matches[i].distance <= max(2 * min_dist, 20.0)) {//ֻ�������������С�ĵ�
			good_matches.push_back(matches[i]);
		}
	}

	cout << good_matches.size() << endl;

	//����ƥ����
	Mat outimg, outimg1;
	drawMatches(img1, kp1, img2, kp2, matches, outimg);
	drawMatches(img1, kp1, img2, kp2, good_matches, outimg1);

	imshow("δɸѡ���", outimg);
	imshow("ɸѡ���", outimg1);
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
		if (matches[i].distance <= max(2 * min_dist, 20.0)) {//ֻ�������������С�ĵ�
			good_min.push_back(matches[i]);
		}
	}
}

//RANSAC�㷨ʵ��
void ransac(vector<DMatch>& matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint, vector<DMatch>& match_ransac) {
	//���屣��ƥ��������
	vector<Point2f> srcPoints(matches.size()), dstPoints(matches.size());

	//����ӹؼ�������ȡ����ƥ���Ե�����
	for (int i = 0; i < matches.size(); i++) {
		srcPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
		//srcPoints��һ���洢����������������飬queryKeyPoint�ǲ�ѯͼ���е������㼯�ϣ�matches��ƥ������������
		//matches[i].queryIdx��ʾƥ����еĵ�i��ƥ��ԵĲ�ѯͼ���е������������
		dstPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;//һ�仰--�������㼯�ϲ���ƥ��Եĵ�
	}

	vector<int> inlinersMask(srcPoints.size());
	findHomography(srcPoints, dstPoints, RANSAC, 5, inlinersMask);//��������ͼ��֮��ĵ�Ӧ�Ծ���  bug
	//���������������srcPoints����ѯͼ���е����������꣩��dstPoints���ο�ͼ���е����������꣩��method�����㵥Ӧ�Ծ���ķ���
	//RANSAC��һ�ֳ����ķ�������ransacReprojThreshold��RANSAC�㷨�е���ֵ����mask����������룬��ʾ��Щ�㱻��Ϊ���ڵ㣩�ȡ�
	//RANSAC��������Ч���ų�����ƥ��������㣬�Ӷ��õ�����׼ȷ�ĵ�Ӧ�Ծ���
	//inlinersMask�е�ÿ��Ԫ�ض�������Ϊ0��1������0��ʾ��Ӧ������������㣨�������ϵ�Ӧ�Ա任ģ�ͣ�����1��ʾ��Ӧ�����������ڵ㣨�����ϵ�Ӧ�Ա任ģ�ͣ���

	//ת��DMatch��ʽ
	for (int i = 0; i < inlinersMask.size(); i++) {
		if (inlinersMask[i]) {//ֻ��Ϊ1�Ĳ�push
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

	//����������
	orb_features(img1, kp1, des1);
	orb_features(img2, kp2, des2);

	//������ƥ��
	vector<DMatch> matches, good_min, good_ransac;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(des1, des2, matches);
	cout << matches.size() << endl;

	//����ɸѡ
	match_min(matches, good_min);
	cout << good_min.size() << endl;//0??? fished

	//ransac�㷨ɸѡ
	ransac(good_min, kp1, kp2, good_ransac);

	Mat outimg, outimg1, outimg2;
	drawMatches(img1, kp1, img2, kp2, matches, outimg);
	drawMatches(img1, kp1, img2, kp2, good_min,outimg1);
	drawMatches(img1, kp1, img2, kp2, good_ransac, outimg2);

	//imshow("δɸѡ", outimg);
	//imshow("��С����ɸѡ", outimg1);
	//imshow("ransacɸѡ", outimg2);
	waitKey(0);
}

void Demo::Demo55_Camera() {//���ģ��
	Mat cameraMatrix;//�ڲξ��� f dx dy
	Mat disCoeffs;//������� k1 k2 k3 p1 p2

	Mat rvec;//��ת  ע����Mat����
	Mat tvec;//ƽ��  ������ת���������ϵ

	vector<Point3f> PointSets;//�ռ�ĵ� 3d
	vector<Point2f> imagePoints;//ӳ�䵽���ؿռ�ĵ�
	projectPoints(PointSets, rvec, tvec, cameraMatrix, disCoeffs, imagePoints);
}

void Demo::Demo56_CameraFind() {//����궨
	vector<Mat> imgs;//�궨��ͼƬ
	string imageName;
	ifstream fin("F:/test-picture/cailbdata.txt");
	while (getline(fin, imageName)) {//��ӱ궨��ͼƬ
		Mat img = imread(imageName);
		imgs.push_back(img);
	}

	Size board_size = Size(9, 6);//�궨���ڽǵ���Ŀ���� �У�
	vector<vector<Point2f>> imgsPoints;//ÿһ��ͼƬ�Ľǵ����꣨���ţ�
	for (int i = 0; i < imgs.size(); i++) {
		Mat img1 = imgs[i];
		Mat gray1;
		cvtColor(img1, gray1, COLOR_BGR2GRAY);
		vector<Point2f> img1_points;
		findChessboardCorners(gray1, board_size, img1_points);//���ͼ�������̸�ģʽ�Ľǵ�
		find4QuadCornerSubpix(gray1, img1_points, Size(5, 5));//�Գ�ʼ�Ľǵ�������������ؼ�����Ż�
		bool pattern = true;
		drawChessboardCorners(img1, board_size, img1_points, pattern);//���Ƽ�⵽�����̸�ǵ�
		imshow("img1", img1);
		waitKey(0);
		imgsPoints.push_back(img1_points);
	}

	Size squareSize = Size(10, 10);//�������̸���ÿ��������ʵ�ߴ�
	vector<vector<Point3f>> objectPoints;//��ʵ��������
	for (int i = 0; i < imgsPoints.size(); i++) {
		vector<Point3f> tempPointSet;
		for (int j = 0; j < board_size.height; j++) {//Ϊÿ���ǵ���������
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
	
	//ͼ��ߴ�
	Size imageSize;
	imageSize.width = imgs[0].cols;
	imageSize.height = imgs[0].rows;

	Mat cameraMatrix = Mat(Size(3, 3), CV_32FC1);//�ڲξ���
	Mat distCoeffs = Mat(1, 5, CV_32FC1);//������� k1 k12 k3 p1 p2
	vector<Mat> rvecs;//ÿ��ͼ����ת����  ÿ��ͼ���ӦΨһ����ξ���
	vector<Mat> tvecs;//ÿ��ͼ��ƽ����
	calibrateCamera(objectPoints, imgsPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);//��ȡ�ڲξ���ͻ������
}

//�Զ���ͼƬ��������
void undist(vector<Mat> imgs, Mat cameraMatrix, Mat distCoeffs, vector<Mat>& undistImgs) {
	for (int i = 0; i < imgs.size(); i++) {
		Mat undistImg;
		undistort(imgs[i], undistImg, cameraMatrix, distCoeffs);
		undistImgs.push_back(undistImg);
	}
}

void Demo::Demo57_Camera() {//ͼ�����
	//undist();


}

void Demo::Demo58_Pnp() {//��Ŀλ�˹��� ��ȷ���������������ϵ�е�λ�úͳ���
	Mat img1,gray1;
	Size board_size = Size(9, 6);//�궨���ڽǵ���Ŀ���� �У�
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	vector<Point2f> imgPoints;//�ǵ�
	findChessboardCorners(gray1, board_size, imgPoints);//���ͼ�������̸�ģʽ�Ľǵ�
	find4QuadCornerSubpix(gray1, imgPoints, Size(5, 5));//�Գ�ʼ�Ľǵ�������������ؼ�����Ż�

	vector<Point3f> PointSets;//�ǵ��Ӧ����������ϵλ��(��һ��λ���������ϵ�ĵ�)
	Mat cameraMatrix = Mat(Size(3, 3), CV_32FC1);//�ڲξ���
	Mat distCoeffs = Mat(1, 5, CV_32FC1);//������� k1 k12 k3 p1 p2
	Mat rvec, tvec;
	solvePnP(PointSets, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);//�����ת������ƽ������ 

	Mat R;
	Rodrigues(rvec, R);//��ת���������ת����

	//Pnp+Ransac����
	Mat Ranrvec, Rantvec;
	solvePnPRansac(PointSets, imgPoints, cameraMatrix, distCoeffs, Ranrvec, Rantvec);

}

void Demo::Demo59_Absdiff() {//��ֵ������ƶ�����  ��ǰʱ��ͼ��-ǰһʱ��ͼ��/����ͼ��=��������
	VideoCapture capture(0);
	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int numFrame = capture.get(CAP_PROP_FRAME_COUNT);//��֡��

	Mat preFrame, preGray;
	capture.read(preFrame);//��ȡ��һ֡��Ϊ����
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
		absdiff(gray, preGray, binary);//��ǰ֡-����֡��
		threshold(binary, binary, 10, 255,THRESH_BINARY|THRESH_OTSU);
		morphologyEx(binary, binary, 2, k);//������

		imshow("input", frame);
		imshow("result", binary);

		//gray.copyTo(preFrame);//ʹ��ǰһ֡������

		char c = waitKey(5);
		if (c == 27) {
			break;
		}
	}
}

void Demo::Demo60_FlowFarneback() {//���ܹ������������壨��������
	//ǰ����� 1.ͬһ�������Ӧ���������Ȳ���
	//2.��֡ͼ������С���˶�
	//3.�����˶�����һ���ԣ�ĳ��������ÿ�����صı仯����һ�£�
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
		imshow("ԭʼ", nextFrame);
		cvtColor(nextFrame, nextGray, COLOR_BGR2GRAY);

		Mat_<Point2f> flows;//��һ������ÿ��Ԫ�ض���һ��Point2f���͵ĵ㡣
		calcOpticalFlowFarneback(preGray, nextGray, flows, 0.5, 3, 15, 3, 5, 1.2, 0);//��������������˶��ٶ�
		Mat xV = Mat::zeros(preFrame.size(), CV_32FC1);//x�����ٶ�
		Mat yV = Mat::zeros(preFrame.size(), CV_32FC1);//y�����ٶ�

		for (int r = 0; r < flows.rows; r++) {
			for (int c = 0; c < flows.cols; c++) {
				const Point2f& flow_xy = flows.at<Point2f>(r, c);//����Point2f�ĵ�
				//�������ñ�ʾ�����޸ı��󶨵Ķ������ flow_xy ��һ��ֻ�������ã����ڷ��� flows �е�����
				xV.at<float> (r, c)= flow_xy.x;
				yV.at<float>(r, c) = flow_xy.y;
			}
		}

		//���������ĽǶȺͷ�ֵ(�źŻ�ͼ��������ǿ��,��ʾ���źŻ�ͼ����ĳ���ض���Ĵ�С��ǿ��)
		Mat magnitude, angle;//magnitude�洢����õ���������ģ���������еľ��룩,angle�洢����õ��������ĽǶȣ��������еĽǶȣ�
		cartToPolar(xV, yV, magnitude, angle);//���ѿ�������ϵ�µ�����ת��Ϊ������ϵ  ������� x �� y ���������ϳ�һ���������������������ģ�ͽǶ�

		//����ת�ɽǶ���
		angle = angle * 180.0 / CV_PI / 2.0;

		//�ѷ�ֵ��һ����0-255���������ʾ
		normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);

		convertScaleAbs(magnitude, magnitude);//ȡ����ֵ
		convertScaleAbs(angle, angle);

		Mat HSV = Mat::zeros(preFrame.size(), preFrame.type());//��������HSV������
		vector<Mat> result;
		split(HSV, result);
		result[0] = angle;//???  ��ɫ0-180
		result[1] = Scalar(255);//���Ͷ�
		result[2] = magnitude;//����
		merge(result, HSV);

		Mat rgbImg;
		cvtColor(HSV, rgbImg, COLOR_HSV2BGR);

		imshow("�˶������", rgbImg);
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
void Demo::Demo61_FlowPyrLK() {//ϡ�����������
	VideoCapture catpure(0);
	Mat preFrame, prevImg;

	catpure.read(preFrame);
	cvtColor(preFrame, prevImg, COLOR_BGR2GRAY);

	vector<Point2f> Points;//����һ֡�ǵ�
	int maxCorners = 100;//���ǵ����Ŀ
	double quality_level = 0.01;//�����ȼ�
	double minDistance = 10;//�����ǵ�֮�����Сŷʽ����
	int blockSize = 3;
	goodFeaturesToTrack(prevImg, Points, maxCorners, quality_level, minDistance, Mat(), blockSize , false);//false ��ʾ���Ǽ��㺣��˼�ǵ�

	//ϡ����������ز�������
	vector<Point2f> prevPts;//ǰһ֡ͼ��ǵ�����
	vector<Point2f> nextPts;//��ǰ֡ͼ��ǵ�����
	vector<uchar> status;//��⵽��״̬
	vector<float> err;
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);//�����������  Ŀ�꺯���ľ���
	double derivlambda = 0.5;
	int flags = 0;

	//��ʼ��״̬�Ľǵ�
	//vector<Point2f> initPoints;
	//initPoints.insert(initPoints.end(), Points.begin(), Points.end());

	//ǰһ֡ͼ���еĽǵ�����
	prevPts.insert(prevPts.end(), Points.begin(), Points.end());

	while (true)
	{
		Mat nextframe, nextImg;
		if (!catpure.read(nextframe)) {
			break;
		}
		imshow("ԭͼ", nextframe);

		cvtColor(nextframe, nextImg, COLOR_BGR2GRAY);
		//��������
		calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, Size(31, 31), 3, criteria, derivlambda, flags);

		//cout << "status:";//test
		//for (int i = 0; i < status.size(); i++) {
		//	cout << status[i] << " ";
		//}
		//cout << endl;

		//cout << prevPts.size() << endl;//test
		//cout << nextPts.size() << endl;
		//cout << initPoints.size() << endl;

		//���ǵ��Ƿ��ƶ� ���ƶ���ɾ��
		size_t i, k;
		for (i = k = 0; i < nextPts.size(); i++) {//bug...
			//������״̬����
			double dist = abs(prevPts[i].x - nextPts[i].x) + abs(prevPts[i].y - nextPts[i].y);
			if (status[i] && dist > 2) {//��⵽ �Ҿ������ƶ� �š����С�
				prevPts[k] = prevPts[i];
				//initPoints[k] = initPoints[i];
				nextPts[k] = nextPts[i];//bug vectorԽ�磿���� finish
				k++;
				circle(nextframe, nextPts[i], 3, Scalar(0, 255, 0), -1, 8);
			}
		}

		//�����ƶ��ǵ���Ŀ
		prevPts.resize(k);
		nextPts.resize(k);
		//initPoints.resize(k);

		//���Ƹ��ٹ켣
		//draw_lines(nextframe, initPoints, nextPts);
		draw_lines(nextframe, prevPts, nextPts);
		imshow("result", nextframe);

		char c = waitKey(50);
		if (c == 27) {
			break;
		}

		swap(nextPts, prevPts);//��Ҫ��Ϊ�˸���prev  next����ν
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
	Mat img;//����ͼ����5000���������� ÿ����500�� ÿ�����ݴ�Сλ20*20

	//ת�Ҷ�

	Mat images = Mat::zeros(5000,400,CV_8UC1);
	Mat labels = Mat::zeros(5000, 1, CV_8UC1);

	int index = 0;//��¼����ĸ���

	Rect numberImg;//��ÿ��20*20 ת��1*400  ��������imagesý��
	numberImg.x=0;//��ȡ�þ����������Ͻǵ�� x ����
	numberImg.height = 1;
	numberImg.width = 400;

	//images(numberImg);

	for (int r = 0; r < 50; r++) {//10���� ÿ����5��
		int label = r / 5;
		int datay = r * 20;//ÿ���������Ͻ�����
		for (int c = 0; c < 100; c++) {//һ��100��
			int datax = c * 20;
			Mat number = Mat::zeros(Size(20, 20), CV_8UC1);
			for (int x = 0; x < 20; x++) {//����ÿ�����ֵ�20*20����
				for (int y = 0; y < 20; y++) {
					number.at<uchar>(x, y) = img.at<uchar>(datay+x, datax+y);
				}
			}

			//����άͼ��ת��������
			Mat row = number.reshape(1, 1);
			numberImg.y = index;

			//��ӵ�������
			row.copyTo(images(numberImg));
			//images(numberImg) ����ʹ����������������� images �����е�������ͨ�� numberImg ��ָ������Ȥ�ľ�������
			labels.at<uchar>(index, 0) = label;//������
			index++;
		}
	}

	images.convertTo(images, CV_32FC1);//����ת������
	labels.convertTo(labels, CV_32SC1);

	Ptr<ml::TrainData> tdata = ml::TrainData::create(images, ml::ROW_SAMPLE, labels);//ROW_SAMPLEÿһ�д���һ��������Ҳ����ÿһ�а�����һ���������������������� COL_SAMPLEÿ��
	//ָ�� ml::TrainData ���������ָ�� tdata

	//����K������
	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(5);//���������5������
	knn->setIsClassifier(true);//true�������� false�����ع�

	knn->train(tdata);
	//����ѵ�����
	String path;//.yml
	knn->save(path);
	waitKey(0);

	/*******************************************************************************/

	//����ģ�Ͳ�����ѵ��
	Mat datas;
	Mat labels;
	Ptr<KNearest> knn = Algorithm::load<KNearest>("knn_model.yml");

	//����
	Mat result;
	knn->findNearest(datas, 5, result);

	Mat test1;//���ڲ���ͼƬ
	Mat test2;

	resize(test1, test1, Size(20, 20));
	resize(test2, test2, Size(20, 20));
	Mat one = test1.reshape(1, 1);
	Mat tow = test2.reshape(1, 1);

	Mat testdata = Mat::zeros(2, 400, CV_8UC1);
	Rect rect;//��Ϊ�м����
	rect.x = 0;
	rect.y = 0;
	rect.height = 1;
	rect.width = 400;
	
	one.copyTo(testdata(rect));
	rect.y = 1;
	tow.copyTo(testdata(rect));

	//���й���ʶ��
	Mat result2;//2*1 
	knn->findNearest(testdata, 5, result2);
	waitKey(0);
}

//void Demo::Demo63_SVM() {
//	Mat samples, lables;
//	//FileStorage���ڶ�ȡ��д�� XML��YAML �� JSON �ȸ�ʽ���ļ������ڱ���ͼ��ػ���ѧϰģ�͡���������������
//	FileStorage fread("point.yml", FileStorage::READ);//FileStorage::READ ��־��ָʾ���ж�ȡ����
//	fread["data"] >> samples;
//	fread["lables"] >> lables;
//	fread.release();
//
//	vector<Vec3b> colors;
//
//	Mat img;//�����հ�ͼ����ʾ�����
//	Mat img2;
//
//	Ptr<SVM> model = SVM::create();
//
//	model->setKernel(SVM::INTER);//�ں�ģ��
//	model->setType(SVM::C_SVC);//SVM����
//	model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001));
//	//...
//	model->train(TrainData);
//	
//	model->predict();
//}

void Demo::Demo64_Camera() {

}
                                                        
void Demo::Demo65_Net() {//����������ģ��
	string model;
	string config;

	Net net = dnn::readNet(model, config);//����ģ��
	vector<String> layerName = net.getLayerNames();
	for (int i = 0; i < layerName.size(); i++) {
		int ID = net.getLayerId(layerName[i]);
		//��ȡÿ��������Ϣ
		Ptr<Layer> layer = net.getLayer(ID);
		cout << layer->type.c_str() << endl;
	}
}

void Demo::Demo66_Net() {

}