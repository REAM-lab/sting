from dataclasses import dataclass, field

@dataclass(slots=True)
class Variables:
    name: list[str]
    _component: list[str] = field(default_factory=list)

    @property
    def component(self) -> int:
        return self.component
    
    @component.setter
    def component(self, value: list[str]):
        if len(self.name) != len(value):
            raise ValueError("All fields must have the same length.")
        else:
            self._component = value


x = Variables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"])

x.component = ["bus_1", "bus_1", "bus_1"]

y = Variables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=["bus_1", "bus_1", "bus_1"])
print('ok')