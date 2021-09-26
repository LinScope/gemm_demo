
namespace pfnie
{

enum class DeviceType
{
  Cpu,
  Cuda,
  Unknown
};

struct Device
{
  DeviceType type;
  int id;
};

} // namespace pfnie
