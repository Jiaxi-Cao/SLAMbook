/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/frame.h"

namespace myslam {

//初始化列表的方式，直接初始化成员变量，减少内存浪费，更加高效
Frame::Frame(long id, double time_stamp, const Sophus::SE3d &pose, const Mat &left, const Mat &right)
        : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {}


//定义普通帧的ID
Frame::Ptr Frame::CreateFrame() {
    static long factory_id = 0;//保证了每次调用 CreateFrame() 时，都会给新的 Frame 对象分配一个唯一的 ID
    Frame::Ptr new_frame(new Frame);
    new_frame->id_ = factory_id++;//为新的帧分配一个唯一的 ID
    return new_frame;
}

//定义关键帧的ID
void Frame::SetKeyFrame() {
    static long keyframe_factory_id = 0; //定义在类函数里面的静态变量和定义在外面没区别！ 我理解成仅初始化一次
    is_keyframe_ = true;//将当前帧标记为关键帧
    keyframe_id_ = keyframe_factory_id++;////为新的帧分配一个唯一的 ID
}

}
