import { BellIcon } from "lucide-react";
import { Button } from "./ui/button";

export default function NotificationsTrigger() {
  return (
    <Button size="icon-2xl">
      <BellIcon className="size-6" />
    </Button>
  );
}
