// gazebo/gazebo.hh includes a core set of basic gazebo functions
// other files based on needs: gazebo/physics/physics.hh, gazebo/rendering/rendering.hh, 
// or gazebo/sensors/sensors.hh
#include <gazebo/gazebo.hh>

namespace gazebo
{ // each plugin inherits from a plugin type, this case is WorldPlugin class
  class WorldPluginTutorial : public WorldPlugin
  {
    public: WorldPluginTutorial() : WorldPlugin()
            {
              printf("Hello World!\n");
            }
    // mandatory function: Load
    // receives an SDF element that contains the elements and attributes 
    // specified in the loaded SDF File
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
            {
            }
  };
  // plugin must be registered with the simulator using GZ_REGISTER_WORLD_PLUGIN macro
  /*
  other types include
  GZ_REGISTER_MODEL_PLUGIN, GZ_REGISTER_SENSOR_PLUGIN, GZ_REGISTER_GUI_PLUGIN, 
  GZ_REGISTER_SYSTEM_PLUGIN and GZ_REGISTER_VISUAL_PLUGIN.
  */
  GZ_REGISTER_WORLD_PLUGIN(WorldPluginTutorial)
}