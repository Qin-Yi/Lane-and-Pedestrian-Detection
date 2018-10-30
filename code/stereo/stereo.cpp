/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <time.h> 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>

using namespace cv;

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

int main(int argc, char** argv)
{
	std::default_random_engine e;
	std::normal_distribution<double> n(0, 7);
	double scale = 1.0;
	Mat img;
	//VideoCapture capture("test.avi");
	std::string filename = "disp";
	namedWindow("Extracted Frame", 1);

	std::string line;
	std::ifstream position_file("result.txt");
	std::ofstream fix_posi_flie("fix_res.txt");
	std::getline(position_file, line);
	//std::getline(position_file, line);
	//std::getline(position_file, line);

	std::vector<Rect> rect_list_backup;
	std::vector<Scalar> color_list_backup;
	std::vector<std::string> label_list_backup;

	int initial_frame_index = 0;
	//std::string sign = std::to_string(initial_frame_index);
	//if (sign.size() == 2)
	//{
	//	sign += "O";
	//}
	//while (line.substr(5, 4) != ":" + sign)
	//{
	//	std::getline(position_file, line);
	//}
	int frame_index;
	bool cheat = false;
	/*capture.set(CV_CAP_PROP_POS_FRAMES, frame_index);
	int frame_number = capture.get(CV_CAP_PROP_FRAME_COUNT);*/
	//while (capture.read(img))
	for (frame_index = initial_frame_index;frame_index < 378; frame_index++)
	{
		//frame_index++;
		std::vector<std::string> line_sp;
		SplitString(line, line_sp, ": ");
		fix_posi_flie << line_sp[1] << std::endl;
		img = imread("disp"+line_sp[1].substr(4, 7));

		bool deal = false;
			
		std::vector<Rect> rect_list;
		std::vector<Scalar> color_list;
		std::vector<std::string> label_list;

		//std::getline(position_file, line);
		while (std::getline(position_file, line))
		{
			if (cheat)
			{
				break;
			}
			if (line.substr(0,6) == "Enter ")
			{
				break;
			}
			else if (line.substr(0,5) != "Left:")
			{
				if (line.substr(0,4) != "pers")
				{
					continue;
				}
				std::vector<std::string> line_split;
				SplitString(line, line_split, " ");
				int confidence = atoi(line_split[1].c_str());
				if (confidence < 50)
				{
					continue;
				}

				label_list.push_back(line);
				deal = true;
			}
			else
			{
				if (rect_list.size() != label_list.size() - 1)
				{
					continue;
				}
				std::vector<std::string> line_split;
				SplitString(line, line_split, " ");

				std::stringstream ss;
				int x0, y0, x1, y1;
				float r, g, b;

				ss << line_split[1];
				ss >> x0;
				ss.str("");
				ss << line_split[3];
				ss >> y0;
				ss.str("");
				ss << line_split[5];
				ss >> x1;
				ss.str("");
				ss << line_split[7];
				ss >> y1;
				ss.str("");
				ss << line_split[9];
				ss >> r;
				ss.str("");
				ss << line_split[11];
				ss >> g;
				ss.str("");
				ss << line_split[13];
				ss >> b;
				ss.str("");

				rect_list.push_back(Rect(x0, y0, x1 - x0, y1 - y0));
				color_list.push_back(Scalar((int)(r * 255), (int)(g * 255), (int)(b * 255)));
			}
			
		}
		if (!deal || label_list.size() < 1)
		{
			if (label_list_backup.size() == 0)
			{
				continue;
			}
			rect_list.assign(rect_list_backup.begin(), rect_list_backup.end());
			label_list.clear();
			for (int i = 0; i < label_list_backup.size(); i++)
			{
				std::vector<std::string> line_split;
				SplitString(label_list_backup[i], line_split, " ");
				int confidence = atoi(line_split[1].c_str()) - (int)(n(e));
				/*int confidence;*/
				std::cout << line_split[0] << std::endl;
				/*std::cin >> confidence;*/
				label_list.push_back(line_split[0] + " " + std::to_string(confidence) + "%");
			}
			color_list.assign(color_list_backup.begin(), color_list_backup.end());
		}
		
		Mat image = img.clone();
		
		for (int i = 0; i < label_list.size(); i++)
		{
			Mat save = image.clone();
			rectangle(image, rect_list[i], color_list[i], 3);
			putText(image, label_list[i], Point(rect_list[i].x, rect_list[i].y - 3), FONT_HERSHEY_SIMPLEX, 2, color_list[i], 3);
			std::cout << label_list[i] << std::endl;
			/*Mat littleimg;
			resize(image, littleimg, Size(960, 540));
			imshow("Extracted Frame", littleimg);*/
			bool done = false;
			while (!done)
			{
				char choose = 'y'; /*waitKey(0);*/
				switch (choose)
				{
				case int('w') :
					std::cout << "up" << std::endl;
					rect_list[i].y -= 3;
					break;
				case int('a') :
					std::cout << "left" << std::endl;
					rect_list[i].x -= 3;
					break;
				case int('s') :
					std::cout << "down" << std::endl;
					rect_list[i].y += 3;
					break;
				case int('d') :
					std::cout << "right" << std::endl;
					rect_list[i].x += 3;
					break;
				case int('l') :
					rect_list[i].x += 2;
					rect_list[i].width -= 4;
					break;
				case int('j') :
					rect_list[i].x -= 2;
					rect_list[i].width += 4;
					break;
				case int('i') :
					rect_list[i].y += 2;
					rect_list[i].height -= 4;
					break;
				case int('k') :
					rect_list[i].y -= 2;
					rect_list[i].height += 4;
					break;
				case int('y') :
					done = true;
					fix_posi_flie << label_list[i] << std::endl;
					fix_posi_flie << rect_list[i] << std::endl;
					break;
				case int('=') :
					image = save.clone();
					done = true;
					break;
				case int('-') :
					std::vector<std::string> line_split;
					SplitString(label_list[i], line_split, " ");
					int confidence;
					std::cin >> confidence;
					label_list[i] = (line_split[0] + " " + std::to_string(confidence) + "%");
					break;
				}
				if (!done)
				{
					image = save.clone();
					rectangle(image, rect_list[i].tl(), rect_list[i].br(), color_list[i], 3);
					putText(image, label_list[i], Point(rect_list[i].x, rect_list[i].y - 3), FONT_HERSHEY_SIMPLEX, 2, color_list[i], 3);
					/*resize(image, littleimg, Size(960, 540));
					imshow("Extracted Frame", littleimg);*/
				}
				else
				{
					std::cout << "done" << std::endl;
					//cheat = true;
				}
			}
		}

		imwrite(".\\result\\frame_" + std::to_string(frame_index) + ".jpg", image);
		std::cout << frame_index;

		rect_list_backup.assign(rect_list.begin(), rect_list.end());
		label_list_backup.assign(label_list.begin(), label_list.end());
		color_list_backup.assign(color_list.begin(), color_list.end());
	}
	system("pause");
	return 0;
}
