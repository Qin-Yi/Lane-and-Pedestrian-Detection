#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sstream>
#include <iostream>
#include <fstream>

using	namespace std;
using	namespace cv;

int main()
{



	Mat frame,gray,edges,img_hsv;
	frame = imread("lane.jpeg");
	cvtColor(frame, img_hsv, CV_RGB2HSV);
	namedWindow("raw_image", CV_WINDOW_FREERATIO);
	imshow("raw_image", frame);
	//waitKey(0);
	cvtColor(frame, gray, CV_BGR2GRAY);
	namedWindow("gray", CV_WINDOW_FREERATIO);
	imshow("gray", gray);
	//waitKey(0);
	//lower_yellow = np.array([20, 100, 100], dtype = “uint8”)
		//upper_yellow = np.array([30, 255, 255], dtype = ”uint8")
	Vec3i lower_yellow(20, 100, 100);
	Vec3i upper_yellow(30, 255, 255);
	Mat mask_yellow, mask_white, mask_yw, mask_yw_image,blurr_image;
	inRange(img_hsv, lower_yellow, upper_yellow, mask_yellow);
	inRange(gray, 200, 255, mask_white);
	bitwise_or(mask_white, mask_yellow, mask_yw);
	bitwise_and(gray, mask_yw, mask_yw_image);
	int kernel_size = 5;
	GaussianBlur(mask_yw_image, blurr_image, Size(kernel_size, kernel_size), 0, 0);
	int low_threshold = 50;
	int high_threshold = 150;
	Canny(blurr_image, blurr_image, low_threshold, high_threshold, 3);
	Point root_points[1][4];
	root_points[0][0] = Point(440, 320);
	root_points[0][1] = Point(116, 537);
	root_points[0][2] = Point(875, 537);
	root_points[0][3] = Point(544, 326);
	int npt[] = { 4 };
	const Point* ppt[1] = { root_points[0] };
	Mat mask = Mat::zeros(size(frame), CV_8U);
	//fillPoly(mask, ppt, 4, 1, Scalar(255, 255, 255));

	fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255));
	cout << size(mask) << endl;
	bitwise_and(blurr_image, mask, blurr_image);
	vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
	HoughLinesP(blurr_image, lines, 1, CV_PI / 180, 20, 50, 100);

	//【4】依次在图中绘制出每条线段  
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
	}



	namedWindow("final", CV_WINDOW_FREERATIO);
	imshow("final", frame);
	waitKey(0);
	imwrite("final.jpg", blurr_image);
	
}