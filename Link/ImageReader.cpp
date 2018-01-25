//
// Created by Patrick Gemperline on 10/22/17.
//

#include "ImageReader.h"

using namespace std;


ImageReader::ImageReader(){
    inputFile="";
    inputFolder="";
}


ImageReader::ImageReader(String newInputFolder){
    inputFile="";
    inputFolder=newInputFolder;
}


void ImageReader::setInputFile(String newInputFile){
    inputFile=newInputFile;
}

void ImageReader::setInputFolder(String newInputFolder){
    inputFolder=newInputFolder;
}



vector<uchar> ImageReader::readFile()
{

    Mat inputMatrix = imread(inputFile, CV_LOAD_IMAGE_UNCHANGED);

    if (inputMatrix.empty()) //check whether the image is loaded or not
    {
        cout << "Error : Image cannot be loaded..!!" << endl;
        vector<uchar> empty;
        return empty;
    }

    return MatToVector(inputMatrix);

}


vector<vector<uchar> > ImageReader::readFolder(){
    
    if (inputFolder=="") //check whether the image is loaded or not
    {
        cout << "Error : Image cannot be loaded.. (no folder name)!!" << endl;

        vector<vector<uchar> > empty;

        return empty;
    }

    vector < string > fileList = getFileNames(inputFolder);

    int fileCounter=0;

    int numberOfFiles=fileList.size();

    ImageReader temp;

    vector< vector < uchar > > imageVectors;

    while (fileCounter<numberOfFiles){

        temp.setInputFile(fileList[fileCounter]);

        cout << "here" << endl;
        
        imageVectors.push_back(temp.readFile());

        fileCounter++;

    }
    
    return imageVectors;
}






/*
 * Helper Functions
 */
vector<uchar> ImageReader::MatToVector(Mat inputMatrix){

    vector<uchar> returnVector;
    //void* temp;
    if (inputMatrix.isContinuous()) {

        returnVector.assign(inputMatrix.datastart, inputMatrix.dataend);

    }

    else {

        for (int i = 0; i < inputMatrix.rows; ++i) {

            //temp = (returnVector.end(), inputMatrix.ptr<uchar>(i), inputMatrix.ptr<uchar>(i)+inputMatrix.cols);
            returnVector.insert(returnVector.end(), inputMatrix.ptr<uchar>(i), inputMatrix.ptr<uchar>(i)+inputMatrix.cols);

        }

    }

    return returnVector;
}


vector< string > ImageReader::getFileNames(string inputFolder){

    string path= inputFolder;
    
      string callString="./listFiles.sh "+path;
     
      cout<< callString<<endl;
  
      const char* passString=callString.c_str();
      
      system(passString);
  
      cout<<"call worked"<<endl;
  
      std::ifstream inputFile;
  
      string fileNames="fileNames.txt";
  
      inputFile.open(fileNames.c_str());
  
      string inputLine;
      
      vector<string> fileList;
  
      while (inputFile.peek() && !inputFile.eof())
      {
        getline(inputFile,inputLine);
          cout<< inputLine<<endl;
  
        if (inputLine.find(":") != string::npos && 
            inputLine.find("/") != string::npos )
        {
          fileList.push_back(inputLine);
        }
      }
        cout<< fileList.size() <<endl;

             inputFile.close();

      return fileList;
}