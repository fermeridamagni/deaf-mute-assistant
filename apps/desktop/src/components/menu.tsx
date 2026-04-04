import { MenuIcon } from "lucide-react";
import { Button } from "./ui/button";

export default function MenuTrigger() {
  return (
    <Button size="icon-2xl">
      <MenuIcon className="size-6" />
    </Button>
  );
}
