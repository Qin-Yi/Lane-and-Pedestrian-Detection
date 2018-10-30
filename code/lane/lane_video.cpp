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

	/*VideoCapture cap("live_left.avi");
	cv::VideoWriter oVideoWriter_right("E:\lane.avi", CV_FOURCC('M', 'J', 'P', 'G'), 20, cv::Size(2448, 2048), true);
	while (true)
	{
		Mat frame, gray, edges, img_hsv;
		if (!cap.read(frame))
			break;*/
		Mat frame, gray, edges, img_hsv;
		frame = imread("final.jpg");
		cvtColor(frame, img_hsv, CV_RGB2HSV);
		//namedWindow("raw_image", CV_WINDOW_FREERATIO);
		imshow("HSV", img_hsv);
		imwrite("HSV.jpg", img_hsv);
		//waitKey(0);
		cvtColor(frame, gray, CV_BGR2GRAY);
		//namedWindow("gray", CV_WINDOW_FREERATIO);
		//imshow("gray", gray);
		//waitKey(0);
		//lower_yellow = np.array([20, 100, 100], dtype = “uint8”)
		//upper_yellow = np.array([30, 255, 255], dtype = ”uint8")
		Vec3i lower_yellow(30, 0, 0);
		Vec3i upper_yellow(179, 255, 255);
		Mat mask_yellow, mask_white, mask_yw, mask_yw_image, blurr_image;
		inRange(img_hsv, lower_yellow, upper_yellow, mask_yellow);
		//inRange(gray, 200, 255, mask_white);
		//bitwise_or(mask_white, mask_yellow, mask_yw);
		//bitwise_and(gray, mask_yw, mask_yw_image);
		
		/*if ( !oVideoWriter_right.isOpened())
		{
			std::cout << "ERROR: Failed to write the video" << std::endl;
			return -1;
		}*/
		bitwise_and(gray, mask_yellow, mask_yw_image);
		int kernel_size = 5;
		//GaussianBlur(mask_yw_image, blurr_image, Size(kernel_size, kernel_size), 0, 0);
		GaussianBlur(gray, blurr_image, Size(kernel_size, kernel_size), 0, 0);
		int low_threshold = 50;
		int high_threshold = 150;
		Canny(blurr_image, blurr_image, low_threshold, high_threshold, 3);
		Point root_points[1][4];
		root_points[0][0] = Point(1081, 1797);
		root_points[0][1] = Point(453, 2037);
		root_points[0][2] = Point(1861, 2037);
		root_points[0][3] = Point(1689, 1809);
		int npt[] = { 4 };
		const Point* ppt[1] = { root_points[0] };
		Mat mask = Mat::zeros(size(frame), CV_8U);
		//fillPoly(mask, ppt, 4, 1, Scalar(255, 255, 255));

		fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255));
		namedWindow("mask", CV_WINDOW_FREERATIO);
		imshow("mask", mask);
		imwrite("mask.jpg", mask);
		cout << size(mask) << endl;

		bitwise_and(blurr_image, mask, blurr_image);
		namedWindow("after_mask", CV_WINDOW_FREERATIO);
		imshow("after_mask", blurr_image);
		imwrite("after_mask.jpg", blurr_image);

		vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
		HoughLinesP(blurr_image, lines, 1, CV_PI / 180, 20, 50, 100);

		//【4】依次在图中绘制出每条线段  
		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];
			line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255),51, CV_AA);
		}



		namedWindow("final", CV_WINDOW_FREERATIO);
		imshow("final", frame);
		imwrite("final2.jpg", frame);
		waitKey(0);
		//oVideoWriter_right.write(frame);

		
	//}
}