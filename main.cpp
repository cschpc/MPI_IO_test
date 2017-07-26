/*
    Copyright 2017 Fredrik Robertsen

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
    files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
    modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <iostream>
#include <mpi.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
#include <regex>


void startTimer( std::chrono::time_point<std::chrono::system_clock> &start)
{
    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::system_clock::now();
}
void stopTimer(const std::chrono::time_point<std::chrono::system_clock> &start, std::vector<double> &results)
{
    std::chrono::duration<double> elapsed_seconds;
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    results.push_back(elapsed_seconds.count());
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    //
    if (argc == 1)
    {
        std::cout << "usage ./a.out maxSize minSize numFields flush (romio=parameters)" << std::endl;
        exit(0);
    }
    int numFields = atoi(argv[3]);
    int64_t alignment = 0;
    int64_t startOffset = 0;
    int maxSize = atoi(argv[1]);
    int minSize = atoi(argv[2]);
    bool flush = atoi(argv[3]);
    MPI_File fhIndividual;
    MPI_File fhFull;
    MPI_Status status;
    MPI_Offset offset;
    MPI_Datatype localVec;
    MPI_Datatype fieldType;

    int rank;
    int nRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    MPI_Info info;
    MPI_Info_create(&info);
    for(int i = 4; i < argc; i++)
    {
        std::string inp = std::string(argv[i]);
        std::stringstream ss;
        ss.str(inp);
        std::string name;
        std::string value;
        std::getline(ss, name, '=');
        std::getline(ss, value);
        std::cout << name << value << std::endl;
        MPI_Info_set(info, name.c_str(), value.c_str());
    }

    std::mt19937 gen(rank);
    std::uniform_int_distribution<> dis(minSize, maxSize);
        
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    int count = dis(gen);
    int64_t count64 = count;

    int64_t totalSize;
    MPI_Allreduce(&count64, &totalSize, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    
    std::vector<double> outputTime;
    std::vector<double> collective;
    std::vector<double> open;
    std::vector<double> speedV;
    std::vector<double> fileTypeCreate;

    // individual writes
    for(int k = 0; k < 20; k++)
    {
        int *buf = new int[count];   

        double totTime = 0;
        double collectiveTime = 0;
        startTimer(start); 
        
        MPI_File_open(MPI_COMM_WORLD, "fileIndiv", MPI_MODE_CREATE | MPI_MODE_RDWR |MPI_MODE_DELETE_ON_CLOSE , MPI_INFO_NULL, &fhIndividual);
        MPI_File_seek(fhIndividual, offset, MPI_SEEK_SET);
        stopTimer(start, open);
        for(int field = 0; field < numFields; field++)
        {
            for(int i = 0; i < count; i++)
            {
                buf[i] = field*rank;
            }
            // timer start
            startTimer(start); 
            MPI_Offset totalCount = 0;
            MPI_Allreduce(&count64, &totalCount, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    
            MPI_Offset rankOffset = 0;
            MPI_Exscan(&count64, &rankOffset, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            stopTimer(start, collective);

            offset = startOffset + rankOffset;
            offset *= sizeof(int);
            offset += field*totalCount*sizeof(int);

            MPI_Barrier(MPI_COMM_WORLD);
            start = std::chrono::system_clock::now();
            MPI_File_write_at_all(fhIndividual, offset, buf, count, MPI_INT, &status);
            if(flush)
                MPI_File_sync(fhIndividual);
            std::chrono::duration<double> elapsed_seconds;
            std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;   
            totTime += elapsed_seconds.count(); 
            // timer stop
        }
        MPI_File_close(&fhIndividual);
        double speed = static_cast<double>(count*numFields*sizeof(int))/1000.0/1000.0/ (totTime);
        double totalSpeed;
        MPI_Allreduce(&speed, &totalSpeed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        speedV.push_back(totalSpeed);
        MPI_Barrier(MPI_COMM_WORLD);
        delete[] buf;
    }
    if(rank == 0)
    {
        std::cout << rank << " Writing fields individually, " << totalSize*numFields*sizeof(int) << " bytes " << std::endl;
        std::cout << "max " << *std::max_element(speedV.begin(), speedV.end()) << " MBytes/s" << std::endl;
        std::cout << "min " << *std::min_element(speedV.begin(), speedV.end()) << " MBytes/s" << std::endl;
        std::cout << "Collective time max: " << *std::max_element(collective.begin(), collective.end()) << " s" << std::endl;
        std::cout << "Collective time min: " << *std::min_element(collective.begin(), collective.end()) << " s" << std::endl;
        std::cout << "File open time max: " << *std::max_element(open.begin(), open.end()) << " s" << std::endl;
        std::cout << "File open time min: " << *std::min_element(open.begin(), open.end()) << " s" << std::endl;
    }
    
    speedV.clear();
    collective.clear();
    open.clear();
    outputTime.clear();
    
    // all in one go
    for(int k = 0; k < 20; k++)
    {
        int *bufFull = new int[count*numFields]; 

        int j = 0;
        for(int field = 0; field < numFields; field++)
        {
            for(int i = 0; i < count; i++)
            {
                bufFull[j] = field*rank;
                j++;
            }
        }
        // timer start
        startTimer(start); 

        int64_t totalCount = 0;
        MPI_Allreduce(&count64, &totalCount, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    
        int64_t rankOffset = 0;
        MPI_Exscan(&count64, &rankOffset, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        stopTimer(start, collective);
    
        offset = startOffset + rankOffset;
        offset *= sizeof(int);

        int64_t temp = numFields*count;
        startTimer(start); 
        
        MPI_File_open(MPI_COMM_WORLD, "fileAll", MPI_MODE_CREATE | MPI_MODE_RDWR |MPI_MODE_DELETE_ON_CLOSE , MPI_INFO_NULL, &fhFull);
        MPI_File_seek(fhFull, offset, MPI_SEEK_SET);
        stopTimer(start, open);
        startTimer(start); 
        
        MPI_Type_contiguous(count, MPI_INT, &fieldType);
        MPI_Type_commit(&fieldType);
        int blockLen[numFields];
        MPI_Aint displacements[numFields];
        for(int i = 0;i < numFields; i++)
        {
            blockLen[i] = count;
            displacements[i] = totalCount*i*sizeof(int);
        }
        MPI_Type_create_hindexed(numFields, blockLen, displacements, MPI_INT, &localVec);
        MPI_Type_commit(&localVec);
        stopTimer(start, fileTypeCreate);

        startTimer(start);         

        MPI_File_set_view(fhFull, offset, MPI_INT, localVec, "native" , MPI_INFO_NULL);
        MPI_File_write_at_all(fhFull, 0, bufFull, numFields, fieldType, &status);
        if(flush)        
            MPI_File_sync(fhFull);
        stopTimer(start, outputTime);
        // timer stop

        MPI_File_close(&fhFull);

        double speed = static_cast<double>(count*numFields*sizeof(int))/1000.0/1000.0/outputTime.back();
        double totalSpeed;
        MPI_Allreduce(&speed, &totalSpeed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        speedV.push_back(totalSpeed);
        delete[] bufFull;
    }
    

    MPI_Finalize();
    if(rank == 0)
    {
        std::cout << rank << " Writing fields individually, " << totalSize*numFields*sizeof(int) << " bytes " << std::endl;
        std::cout << "max " << *std::max_element(speedV.begin(), speedV.end()) << " MBytes/s" << std::endl;
        std::cout << "min " << *std::min_element(speedV.begin(), speedV.end()) << " MBytes/s" << std::endl;
        std::cout << "Collective time max: " << *std::max_element(collective.begin(), collective.end()) << " s" << std::endl;
        std::cout << "Collective time min: " << *std::min_element(collective.begin(), collective.end()) << " s" << std::endl;
        std::cout << "File open time max: " << *std::max_element(open.begin(), open.end()) << " s" << std::endl;
        std::cout << "File open time min: " << *std::min_element(open.begin(), open.end()) << " s" << std::endl;
    }
    exit(0);
}