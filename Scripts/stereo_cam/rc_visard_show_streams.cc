#include <rc_visard_opencv_example/gc_cleaner.h>
#include <rc_visard_opencv_example/gc_receiver.h>

#include <rc_genicam_api/config.h>

#if CV_MAJOR_VERSION == 2
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#else
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <iostream>
#include <csignal>
#include <atomic>
#include <map>
#include <fstream>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

namespace ipc = boost::interprocess;

namespace
{
std::atomic_bool stop;

void signal_handler(int sig)
{
  signal(sig, signal_handler);
  stop = true;
}
}

// Struct to store image metadata in shared memory
struct SharedImageData {
    int rows;
    int cols;
    int type;
    bool new_data;
    char stream_name[32];
    // Actual image data will follow this struct in memory
};

bool left = false;
bool right = false;
bool disparity = false;
bool confidence = false;
bool error = false;

bool synchronize_data = false;
bool ipc_mode = false;
std::string shared_mem_name = "rc_visard_shared_mem";
std::string mutex_name = "rc_visard_mutex";

int frame_rate = 0;

// command line flags
std::map<std::string, bool *> flags
    {
        {"--left",        &left},
        {"--right",       &right},
        {"--disparity",   &disparity},
        {"--confidence",  &confidence},
        {"--error",       &error},
        {"--synchronize", &synchronize_data},
        {"--ipc",         &ipc_mode}
    };

std::string device;

// parse command line arguments
bool parseArguments(int argc, char **argv)
{
  if (argc < 2)
  { return false; }

  // iterate command line flags and check if each is contained in the flags map
  for (int i = 1; i < argc - 1; ++i)
  {
    const auto flag = flags.find(argv[i]);
    if (flag != flags.end())
    {
      *flag->second = true;
    }
    else if (std::string(argv[i]).find("--frame-rate=") == 0)
    {
      const std::string f_str = std::string(argv[i]).substr(13);
      try
      {
        frame_rate = std::stoi(f_str);
      }
      catch (const std::exception &)
      {
        std::cerr << "Value of frame rate is no number" << std::endl;
        return false;
      }
    }
    else if (std::string(argv[i]).find("--shared-mem-name=") == 0)
    {
      shared_mem_name = std::string(argv[i]).substr(18);
    }
    else if (std::string(argv[i]).find("--mutex-name=") == 0)
    {
      mutex_name = std::string(argv[i]).substr(13);
    }
    else
    {
      std::cerr << "Argument " << argv[i] << " not known" << std::endl;
      return false;
    }
  }

  // read device id, it should be at the last position
  const std::string dev = argv[argc - 1];
  if (dev.find("--") == 0)
  {
    std::cerr << "Final argument must be device ID" << std::endl;
    return false;
  }

  device = dev;
  return true;
}

// name of the OpenCV window
static const std::string cv_win_name = "stream";

// Function to write image data to shared memory
void writeToSharedMemory(const cv::Mat& img, const std::string& stream_name) {
    try {
        // Calculate required size for the shared memory
        size_t data_size = img.total() * img.elemSize();
        size_t total_size = sizeof(SharedImageData) + data_size;
        
        // Create stream-specific shared memory name
        std::string stream_shm_name = shared_mem_name + "_" + stream_name;
        std::string stream_mutex_name = mutex_name + "_" + stream_name;
        
        // Create or open shared memory object
        ipc::shared_memory_object shm(
            ipc::open_or_create,
            stream_shm_name.c_str(),
            ipc::read_write
        );
        
        // Set the size of the shared memory
        shm.truncate(total_size);
        
        // Map the whole shared memory in this process
        ipc::mapped_region region(shm, ipc::read_write);
        
        // Get the address of the mapped region
        void* addr = region.get_address();
        
        // Placement new to create the SharedImageData at the start of the mapped region
        SharedImageData* shared_data = new (addr) SharedImageData;
        
        // Fill the shared data structure
        shared_data->rows = img.rows;
        shared_data->cols = img.cols;
        shared_data->type = img.type();
        strncpy(shared_data->stream_name, stream_name.c_str(), sizeof(shared_data->stream_name) - 1);
        shared_data->stream_name[sizeof(shared_data->stream_name) - 1] = '\0';
        
        // Create a named mutex for synchronization
        ipc::named_mutex mutex(ipc::open_or_create, stream_mutex_name.c_str());
        
        // Lock the mutex before writing to shared memory
        mutex.lock();
        
        // Copy the image data to the shared memory (after the SharedImageData structure)
        char* data_addr = static_cast<char*>(addr) + sizeof(SharedImageData);
        memcpy(data_addr, img.data, data_size);
        
        // Mark as new data
        shared_data->new_data = true;
        
        // Unlock the mutex
        mutex.unlock();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error writing to shared memory: " << ex.what() << std::endl;
    }
}

int main(int argc, char **argv)
{
  // install signal handler to catch Ctrl+C
  signal(SIGINT, signal_handler);

  if (!parseArguments(argc, argv))
  {
    std::cerr << "Usage: " << argv[0] << " [options] <device id>\n";
    for (const auto &s : flags)
    {
      std::cerr << '\t' << s.first << '\n';
    }
    std::cerr << '\t' << "--frame-rate=<n>" << '\n';
    std::cerr << '\t' << "--shared-mem-name=<name>" << '\n';
    std::cerr << '\t' << "--mutex-name=<name>" << '\n';
    return 1;
  }

  // If IPC mode is enabled, clean up any existing shared memory
  if (ipc_mode) {
    try {
      // Remove the base shared memory if it exists
      ipc::shared_memory_object::remove(shared_mem_name.c_str());
      ipc::named_mutex::remove(mutex_name.c_str());
      
      // Remove all stream-specific shared memory segments
      std::vector<std::string> streams = {"left", "right", "disparity", "confidence", "error"};
      for (const auto& stream : streams) {
        std::string stream_shm_name = shared_mem_name + "_" + stream;
        std::string stream_mutex_name = mutex_name + "_" + stream;
        ipc::shared_memory_object::remove(stream_shm_name.c_str());
        ipc::named_mutex::remove(stream_mutex_name.c_str());
      }
      
      std::cout << "IPC mode enabled with shared memory base name: " << shared_mem_name << std::endl;
    }
    catch (const std::exception& ex) {
      // Ignore if it doesn't exist
    }
  }

  // RAII genicam resource cleaner
  GcCleaner gc_cleaner;

  // wrapper around rc_genicam_api
  GcReceiver gc_receiver(device, synchronize_data);

  // open connection to device
  if (!gc_receiver.open())
  {
    std::cerr << "Could not open device '" << device << '\'' << std::endl;
    return 1;
  }

  // depending on command line arguments,
  // create respective image receiver factories
  GcReceiver::ReceiverFactories receiver_factories;

  if (left || right)
  {
    LeftRight left_right;
    if (left && right)
    { left_right = LeftRight::LEFT_RIGHT; }
    else if (left)
    { left_right = LeftRight::LEFT; }
    else
    { left_right = LeftRight::RIGHT; }
    receiver_factories.insert(std::make_shared<IntensityReceiverFactory>(
        left_right, true));
  }
  if (disparity)
  {
    receiver_factories.insert(std::make_shared<DisparityReceiverFactory>());
  }
  if (confidence)
  {
    receiver_factories.insert(std::make_shared<ConfidenceReceiverFactory>());
  }
  if (error)
  {
    receiver_factories.insert(std::make_shared<ErrorReceiverFactory>());
  }

  if (receiver_factories.empty())
  {
    std::cerr << "At least one stream must be enabled" << std::endl;
    return 1;
  }

  // enable streams and start streaming
  if (!gc_receiver.initializeStreams(receiver_factories))
  {
    std::cerr << "Could not initialize flags" << std::endl;
    return 1;
  }

  // set frame rate if requested
  if (frame_rate > 0)
  {
    if (!rcg::setFloat(gc_receiver.getNodeMap(), "AcquisitionFrameRate",
                       frame_rate, false))
    {
      std::cerr << "Could not set frame rate" << std::endl;
      return 1;
    }
  }

  std::cout
      << "Press 'n'(ext) or 'p'(revious) to cycle through streams, 'q' to exit"
      << std::endl;

  // Put all enabled streams in a list to create a mapping from some continuous
  // index to the streams. This index will be used to cycle through the streams.
  std::vector<bool *> show;
  if (left)
  { show.push_back(&left); }
  if (right)
  { show.push_back(&right); }
  if (disparity)
  { show.push_back(&disparity); }
  if (confidence)
  { show.push_back(&confidence); }
  if (error)
  { show.push_back(&error); }

  int current_stream = 0;

  // Only create OpenCV window if not in IPC mode
  if (!ipc_mode) {
    cv::namedWindow(cv_win_name, cv::WINDOW_NORMAL);
  }

  // Method for setting the OpenCV window title.
  // This is only available beginning with OpenCV 3.0
  auto setWindowTitle = [](const std::string &title)
  {
#if CV_MAJOR_VERSION >= 3
    if (!ipc_mode) {
      cv::setWindowTitle(cv_win_name, title);
    }
#endif
  };

  // loop until Ctrl+C is hit
  while (!stop)
  {
    // receive images with a 3 s timeout
    const auto image_set = gc_receiver.receive(std::chrono::seconds(3));
    if (!image_set)
    {
      std::cerr << "Did not receive data before timeout" << std::endl;
      continue;
    }

    if (ipc_mode) {
      // Write all available streams to shared memory
      if (image_set->left_img_) {
          writeToSharedMemory(image_set->left_img_->data_, "left");
      }
      if (image_set->right_img_) {
          writeToSharedMemory(image_set->right_img_->data_, "right");
      }
      if (image_set->disparity_img_) {
          cv::Mat disp = image_set->disparity_img_->data_;
          cv::threshold(disp, disp, 1000, 0, cv::THRESH_TOZERO_INV);
          cv::normalize(disp, disp, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
          writeToSharedMemory(disp, "disparity");
      }
      if (image_set->confidence_img_) {
          cv::Mat conf = image_set->confidence_img_->data_;
          cv::normalize(conf, conf, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
          writeToSharedMemory(conf, "confidence");
      }
      if (image_set->error_img_) {
          cv::Mat err = image_set->error_img_->data_;
          cv::normalize(err, err, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
          writeToSharedMemory(err, "error");
      }
    } else {
      // Original display code for non-IPC mode
      if (show[current_stream] == &left && image_set->left_img_) {
          cv::imshow(cv_win_name, image_set->left_img_->data_);
          setWindowTitle("left");
      }
      if (show[current_stream] == &right && image_set->right_img_) {
          cv::imshow(cv_win_name, image_set->right_img_->data_);
          setWindowTitle("right");
      }
      if (show[current_stream] == &disparity && image_set->disparity_img_) {
          cv::Mat disp = image_set->disparity_img_->data_;
          cv::threshold(disp, disp, 1000, 0, cv::THRESH_TOZERO_INV);
          cv::normalize(disp, disp, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
          cv::imshow(cv_win_name, disp);
          setWindowTitle("disparity");
      }
      if (show[current_stream] == &confidence && image_set->confidence_img_) {
          cv::Mat conf = image_set->confidence_img_->data_;
          cv::normalize(conf, conf, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
          cv::imshow(cv_win_name, conf);
          setWindowTitle("confidence");
      }
      if (show[current_stream] == &error && image_set->error_img_) {
          cv::Mat err = image_set->error_img_->data_;
          cv::normalize(err, err, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
          cv::imshow(cv_win_name, err);
          setWindowTitle("error");
      }
    }

    // read key presses (only in non-IPC mode)
    if (!ipc_mode) {
      const int key = cv::waitKey(1);
      if (key > 0)
      {
        const char ascii_key = static_cast<char>(key);
        if (ascii_key == 'n')
        {
          // go to next stream
          current_stream = (current_stream + 1) % static_cast<int>(show.size());
        }
        if (ascii_key == 'p')
        {
          // go to previous stream
          current_stream = (current_stream - 1) % static_cast<int>(show.size());
        }
        if (ascii_key == 'q')
        {
          // quit
          break;
        }
      }
    }
  }

  // Clean up shared memory resources
  if (ipc_mode) {
    // Remove all stream-specific shared memory segments
    std::vector<std::string> streams = {"left", "right", "disparity", "confidence", "error"};
    for (const auto& stream : streams) {
        std::string stream_shm_name = shared_mem_name + "_" + stream;
        std::string stream_mutex_name = mutex_name + "_" + stream;
        ipc::shared_memory_object::remove(stream_shm_name.c_str());
        ipc::named_mutex::remove(stream_mutex_name.c_str());
    }
    
    // Also remove the base shared memory if it exists
    ipc::shared_memory_object::remove(shared_mem_name.c_str());
    ipc::named_mutex::remove(mutex_name.c_str());
  }
  
  return 0;
}