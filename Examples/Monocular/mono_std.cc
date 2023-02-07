/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>
#include"System.h"
#include "opencv2/core/eigen.hpp"

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

std::string img_path;
std::string txt_path;

struct KittiPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KittiPose(const cv::Mat &Tcw)
    {
        Eigen::Matrix4d Tcw_E;
        if (Tcw.empty()){
            Tcw_E = Eigen::Matrix4d::Identity();
        }else{
            cv::cv2eigen(Tcw, Tcw_E);
        }
        Eigen::Vector3d tcw = Tcw_E.block<3, 1>(0, 3);
        Eigen::Matrix3d Rcw = Tcw_E.block<3, 3>(0, 0);
        Eigen::Vector3d twc = - Rcw.inverse() * tcw;
        Eigen::Matrix3d Rwc = Rcw.inverse();
        Twc_[3] = twc.x();
        Twc_[7] = twc.y();
        Twc_[11] = twc.z();

        for( int i = 0 ; i < 3 ; i++ ) {
            Twc_[i] = Rwc(0,i);
            Twc_[i+4] = Rwc(1,i);
            Twc_[i+8] = Rwc(2,i);
        }
    }

    double Twc_[12];
};
std::vector<KittiPose> vkittipose_;
void writeTrajectoryKITTI(const std::string &filename)
{
    std::ofstream f;

    std::cout << "\n Going to write the computed trajectory into : " << filename << "\n";

    f.open(filename.c_str());
    f << std::fixed;

    size_t nbposes = vkittipose_.size();
    for( size_t i = 0 ; i < nbposes ; i++ )
    {
        double *T = vkittipose_.at(i).Twc_;

        f << std::setprecision(9)
          << T[0] << " " << T[1] << " " << T[2] << " " << T[3] << " "
          << T[4] << " " << T[5] << " " << T[6] << " " << T[7] << " "
          << T[8] << " " << T[9] << " " << T[10] << " " << T[11]
          << std::endl;

        f.flush();
    }

    f.close();

    std::cout << "\nKITTITrajectory file written!\n";
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }
    img_path = argv[3];
    txt_path = argv[4];
    std::cout<<" load img from "<<img_path<<std::endl;
    std::cout<<" save txt to "<<txt_path<<std::endl;

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;

    LoadImages(img_path, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(float ni=0; ni<nImages; )
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[(int)ni],CV_8UC1);
        cv::resize(im, im, cv::Size(640, 360));
        double tframe = vTimestamps[(int)ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[(int)ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        cv::Mat Tcw = SLAM.TrackMonocular(im,tframe);
        vkittipose_.emplace_back(Tcw);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        {
            double cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout<<"cost "<<cost_ms<<"  ms id: "<<ni<<std::endl;
            int hz = 50;
            ni = ni + std::max(1, (int)((double)cost_ms * (double)hz / (double)1000));
            if (ni >= nImages)
                break;
        }

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
        //usleep(0.2*1e6);
    }
    writeTrajectoryKITTI("kitti.txt");
    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{

    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    string strPrefixLeft = strPathToSequence + "/image/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(5) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}
