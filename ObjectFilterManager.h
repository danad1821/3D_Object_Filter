#ifndef OBJECT_FILTER_MANAGER_H
#define OBJECT_FILTER_MANAGER_H

#include <map>
#include <memory>   
#include <string>
#include <mutex>
#include "ObjectPoseEMAFilter.h"
namespace ORB_SLAM3 {
class ObjectFilterManager {
    public:
        ObjectFilterManager(const std::string &strSettingPath = "", bool filterActive = true);

        ObjectPoseEMAFilter& getOrCreateFilter(int objId);

        Eigen::Matrix4f filterObjectPose(int objId, const Eigen::Matrix4f& rawPose, float scale);

        void removeFilter(int objId);

        bool hasFilter(int objId) const;

        void setSettingStr(const std::string &strSettingPath);

        std::string getSettingStr();

    private:
        std::map<int, std::unique_ptr<ObjectPoseEMAFilter>> mObjectFilters;

        std::string mStrSettingPath;

        bool mFilterActive; // Flag to enable/disable the filter

        std::mutex mMutexFilters; // Mutex to protect access to the filters map

};
}

#endif // OBJECT_FILTER_MANAGER_H